/*
 *  Copyright 2011-2015 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "network_schema.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"
#include "data_layer.h"
#include "color_palette.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <list>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/smallest_last_ordering.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/reverse_graph.hpp>

namespace nnforge
{
	network_schema::network_schema()
	{
	}

	network_schema::network_schema(const std::vector<layer::const_ptr>& layer_list)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
			add_layer(*it);
		fill_name_map();
		add_edges();
		detect_cycles();
	}
	
	/*
	std::vector<layer_data_configuration_list> network_schema::get_layer_data_configuration_list_list() const
	{
		std::vector<layer_data_configuration_list> res;

		for(const_layer_list::const_iterator it = layers.begin(); it != layers.end(); ++it)
			res.push_back((*it)->get_layer_data_configuration_list());

		return res;
	}
*/
	void network_schema::write_proto(std::ostream& stream_to_write_to) const
	{
		protobuf::NetworkSchema schema;
		if (!name.empty())
			schema.set_name(name);

		for (std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
		{
			layer::const_ptr current_layer = layers[*vp.first].l;
			protobuf::Layer * layer_proto = schema.add_layer();
			layer_proto->set_type(current_layer->get_type_name());
			if (!current_layer->instance_name.empty())
				layer_proto->set_name(current_layer->instance_name);
			for(std::vector<std::string>::const_iterator it = current_layer->input_layer_instance_names.begin(); it != current_layer->input_layer_instance_names.end(); ++it)
				*layer_proto->add_input_layer_name() = *it;

			current_layer->write_proto(layer_proto);
		}

		google::protobuf::io::OstreamOutputStream output_stream(&stream_to_write_to);
		google::protobuf::TextFormat::Print(schema, &output_stream);
	}

	void network_schema::read_proto(std::istream& stream_to_read_from)
	{
		clear();

		protobuf::NetworkSchema schema;
		google::protobuf::io::IstreamInputStream input_stream(&stream_to_read_from);
		google::protobuf::TextFormat::Parse(&input_stream, &schema);

		name = schema.name();

		for(int i = 0; i < schema.layer_size(); ++i)
		{
			const nnforge::protobuf::Layer& src_layer = schema.layer(i);
			layer::ptr new_layer = single_layer_factory::get_const_instance().create_layer(src_layer.type());
			new_layer->instance_name = src_layer.name();
			for(int j = 0; j < src_layer.input_layer_name_size(); ++j)
				new_layer->input_layer_instance_names.push_back(src_layer.input_layer_name(j));
			new_layer->read_proto(&src_layer);
			add_layer(new_layer);
		}

		fill_name_map();
		add_edges();
		detect_cycles();
	}

	void network_schema::add_layer(layer::const_ptr new_layer)
	{
		schema_graph::vertex_descriptor new_layer_descriptor = boost::add_vertex(layers);
		layers[new_layer_descriptor].l = new_layer;
	}

	void network_schema::fill_name_map()
	{
		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
		{
			schema_graph::vertex_descriptor layer_descriptor = *vp.first;
			layer::const_ptr l = layers[layer_descriptor].l;
			if (!layer_instance_name_to_vertex_decriptor_map.insert(std::make_pair(l->instance_name, layer_descriptor)).second)
				throw neural_network_exception((boost::format("Duplicate layer name: %1%") % l->instance_name).str());
		}
	}

	void network_schema::add_edges()
	{
		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
		{
			schema_graph::vertex_descriptor src_vertex = *vp.first;
			const std::vector<std::string>& input_layer_instance_names = layers[src_vertex].l->input_layer_instance_names;
			for(std::vector<std::string>::const_iterator it = input_layer_instance_names.begin(); it != input_layer_instance_names.end(); ++it)
			{
				const std::string& input_layer_instance_name = *it;
				std::map<std::string, schema_graph::vertex_descriptor>::const_iterator name_descriptor_it = layer_instance_name_to_vertex_decriptor_map.find(input_layer_instance_name);
				if (name_descriptor_it == layer_instance_name_to_vertex_decriptor_map.end())
					throw neural_network_exception((boost::format("Unkown input layer name: %1%") % input_layer_instance_name).str());
				schema_graph::vertex_descriptor dst_vertex = name_descriptor_it->second;
				boost::add_edge(src_vertex, dst_vertex, layers);
			}
		}
	}

	void network_schema::clear()
	{
		layers.clear();
		layer_instance_name_to_vertex_decriptor_map.clear();
	}

	void network_schema::detect_cycles()
	{
		bool has_cycle = false;
		cycle_detector vis(has_cycle);
		boost::depth_first_search(layers, boost::visitor(vis));
		if (has_cycle)
			throw neural_network_exception("Cycle in network schema detected");
	}

	std::vector<layer::const_ptr> network_schema::get_required_layers(const std::vector<std::string>& output_layer_names) const
	{
		std::set<schema_graph::vertex_descriptor> visited_vertices;
		record_all_edges<schema_graph::vertex_descriptor> vis(visited_vertices);
		std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));
		for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
		{
			std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it2 = layer_instance_name_to_vertex_decriptor_map.find(*it);
			if (it2 == layer_instance_name_to_vertex_decriptor_map.end())
				throw neural_network_exception((boost::format("Layer not found: %1%") % *it).str());

			boost::depth_first_visit(
				layers,
				it2->second,
				vis,
				boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));
		}

		std::vector<layer::const_ptr> res;
		for(std::set<schema_graph::vertex_descriptor>::const_iterator it = visited_vertices.begin(); it != visited_vertices.end(); ++it)
			res.push_back(layers[*it].l);

		return res;
	}

	std::vector<layer::const_ptr> network_schema::get_required_layers(
		const std::vector<std::string>& output_layer_names,
		const std::vector<std::string>& error_source_layer_names,
		const std::vector<std::string>& exclude_data_update_layer_names) const
	{
		std::set<schema_graph::vertex_descriptor> vertices_with_updatable_weights;
		{
			for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
				if (!layers[*vp.first].l->is_empty_data())
					vertices_with_updatable_weights.insert(*vp.first);
			for(std::vector<std::string>::const_iterator it = exclude_data_update_layer_names.begin(); it != exclude_data_update_layer_names.end(); ++it)
			{
				std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it2 = layer_instance_name_to_vertex_decriptor_map.find(*it);
				if (it2 == layer_instance_name_to_vertex_decriptor_map.end())
					throw neural_network_exception((boost::format("Layer not found: %1%") % *it).str());
				vertices_with_updatable_weights.erase(it2->second);
			}
		}

		std::set<schema_graph::vertex_descriptor> visited_vertices;
		{
			record_all_edges<schema_graph::vertex_descriptor> vis(visited_vertices);
			std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));
			for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
			{
				std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it2 = layer_instance_name_to_vertex_decriptor_map.find(*it);
				if (it2 == layer_instance_name_to_vertex_decriptor_map.end())
					throw neural_network_exception((boost::format("Layer not found: %1%") % *it).str());

				boost::depth_first_visit(
					layers,
					it2->second,
					vis,
					boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));
			}
		}

		for(std::vector<std::string>::const_iterator it = error_source_layer_names.begin(); it != error_source_layer_names.end(); ++it)
		{
			std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it2 = layer_instance_name_to_vertex_decriptor_map.find(*it);
			if (it2 == layer_instance_name_to_vertex_decriptor_map.end())
				throw neural_network_exception((boost::format("Layer not found: %1%") % *it).str());

			std::set<schema_graph::vertex_descriptor> visited_vertices2;
			record_all_edges<schema_graph::vertex_descriptor> vis(visited_vertices2);
			std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));

			boost::depth_first_visit(
				layers,
				it2->second,
				vis,
				boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));

			for(std::set<schema_graph::vertex_descriptor>::const_iterator it2 = visited_vertices2.begin(); it2 != visited_vertices2.end(); ++it2)
			{
				if (vertices_with_updatable_weights.find(*it2) != vertices_with_updatable_weights.end())
				{
					visited_vertices.insert(visited_vertices2.begin(), visited_vertices2.end());
					break;
				}
			}
		}

		std::vector<layer::const_ptr> res;
		for(std::set<schema_graph::vertex_descriptor>::const_iterator it = visited_vertices.begin(); it != visited_vertices.end(); ++it)
			res.push_back(layers[*it].l);

		return res;
	}

	network_action_schema::ptr network_schema::get_actions_for_forward_propagation(const std::vector<std::string>& output_layer_names) const
	{
		network_action_schema::ptr res(new network_action_schema());

		std::vector<layer::const_ptr> layers = get_layers_in_forward_propagation_order();
		for(std::vector<layer::const_ptr>::const_iterator it = layers.begin(); it != layers.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (l->get_type_name() == data_layer::layer_type_name)
				continue;

			std::vector<layer_name_with_action> dependencies;
			for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
			{
				layer::const_ptr dest_layer = get_layer(*it2);
				if (dest_layer->get_type_name() == data_layer::layer_type_name)
					continue;
				dependencies.push_back(layer_name_with_action(*it2, layer_action(layer_action::forward)));
			}

			res->add_action(
				l,
				layer_action(layer_action::forward),
				dependencies);
		}

		return res;
	}

	network_action_schema::ptr network_schema::get_actions_for_backward_propagation(
		const std::vector<std::string>& output_layer_names,
		const std::vector<std::string>& error_source_layer_names,
		const std::vector<std::string>& exclude_data_update_layer_names,
		std::vector<std::vector<layer_name_with_action> >& same_output_action_sets) const
	{
		network_action_schema::ptr res(new network_action_schema());

		same_output_action_sets.clear();

		std::set<layer_name_with_action> target_action_set;
		{
			for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
				target_action_set.insert(layer_name_with_action(*it, layer_action(layer_action::forward)));
		}

		// Add forward prop actions
		{
			std::vector<layer::const_ptr> layers = get_layers_in_forward_propagation_order();
			for(std::vector<layer::const_ptr>::const_iterator it = layers.begin(); it != layers.end(); ++it)
			{
				layer::const_ptr l = *it;
				if (l->get_type_name() == data_layer::layer_type_name)
					continue;

				std::vector<layer_name_with_action> dependencies;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
				{
					layer::const_ptr dest_layer = get_layer(*it2);
					if (dest_layer->get_type_name() == data_layer::layer_type_name)
						continue;
					dependencies.push_back(layer_name_with_action(*it2, layer_action(layer_action::forward)));
				}

				res->add_action(
					l,
					layer_action(layer_action::forward),
					dependencies);
			}
		}

		std::set<std::string> layer_weights_to_update;

		// Add backward actions
		{
			std::set<std::string> exclude_data_update_layer_name_set(exclude_data_update_layer_names.begin(), exclude_data_update_layer_names.end());
			for(std::vector<std::string>::const_iterator it = error_source_layer_names.begin(); it != error_source_layer_names.end(); ++it)
			{
				std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it2 = layer_instance_name_to_vertex_decriptor_map.find(*it);
				if (it2 == layer_instance_name_to_vertex_decriptor_map.end())
					continue; // error source might be dropped now due to it not contributing to weight updates

				std::set<schema_graph::vertex_descriptor> visited_vertices;
				record_all_edges<schema_graph::vertex_descriptor> vis(visited_vertices);
				std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));

				boost::depth_first_visit(
					layers,
					it2->second,
					vis,
					boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));

				for(std::set<schema_graph::vertex_descriptor>::const_iterator it2 = visited_vertices.begin(); it2 != visited_vertices.end(); ++it2)
				{
					layer::const_ptr l = layers[*it2].l;
					if (l->get_type_name() == data_layer::layer_type_name)
						continue;

					for(int backprop_index = 0; backprop_index < l->input_layer_instance_names.size(); ++backprop_index)
					{
						layer_action action(layer_action::backward_data, backprop_index);
						if (!res->action_exists(layer_name_with_action(l->instance_name, action)))
							res->add_action(l, action);
					}

					if ((!l->is_empty_data()) && (exclude_data_update_layer_name_set.find(l->instance_name) == exclude_data_update_layer_name_set.end()))
					{
						layer_action action(layer_action::backward_weights);
						if (!res->action_exists(layer_name_with_action(l->instance_name, action)))
							res->add_action(l, action);
						target_action_set.insert(layer_name_with_action(l->instance_name, action));
						layer_weights_to_update.insert(l->instance_name);
					}
				}
			}
		}

		// Add dependencies 
		{
			std::set<std::string> error_source_layer_name_set(error_source_layer_names.begin(), error_source_layer_names.end());
			for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			{
				layer::const_ptr l = layers[*vp.first].l;
				if (l->get_type_name() == data_layer::layer_type_name)
					continue;
				const std::string& layer_name = l->instance_name;

				std::vector<layer_name_with_action> src_action_list;
				{
					layer_name_with_action weight_update_action(layer_name, layer_action(layer_action::backward_weights));
					if (res->action_exists(weight_update_action))
						src_action_list.push_back(weight_update_action);
					for(int backprop_index = 0; backprop_index < l->input_layer_instance_names.size(); ++backprop_index)
					{
						layer_name_with_action action(layer_name, layer_action(layer_action::backward_data, backprop_index));
						if (res->action_exists(action))
							src_action_list.push_back(action);
					}
				}
				if (src_action_list.empty())
					continue;

				layer_name_with_action dst_action;
				if (error_source_layer_name_set.find(layer_name) == error_source_layer_name_set.end())
				{
					std::vector<layer_name_with_action> dst_action_list;
					for(std::pair<schema_graph::in_edge_iterator, schema_graph::in_edge_iterator> ep = boost::in_edges(*vp.first, layers); ep.first != ep.second; ep.first++)
					{
						schema_graph::vertex_descriptor previous_vertex = boost::source(*ep.first, layers);
						layer::const_ptr pl = layers[previous_vertex].l;
						int backprop_index = 0;
						for(std::vector<std::string>::const_iterator it = pl->input_layer_instance_names.begin(); it != pl->input_layer_instance_names.end(); ++it, ++backprop_index)
						{
							if (*it == layer_name)
							{
								layer_name_with_action action(pl->instance_name, layer_action(layer_action::backward_data, backprop_index));
								if (res->action_exists(action))
									dst_action_list.push_back(action);
								break;
							}
						}
					}
					if (dst_action_list.size() > 1)
					{
						for(int i = 0; i < dst_action_list.size() - 1; ++i)
							res->add_dependency(dst_action_list[i], dst_action_list[i+1]);
						same_output_action_sets.push_back(dst_action_list);
						dst_action_list.resize(1);
					}
					if (dst_action_list.empty())
						continue;
					else
						dst_action = dst_action_list.front();
				}
				else
				{
					dst_action = layer_name_with_action(layer_name, layer_action(layer_action::forward));
					if (!res->action_exists(dst_action))
						continue;
				}

				for(std::vector<layer_name_with_action>::const_iterator src_it = src_action_list.begin(); src_it != src_action_list.end(); ++src_it)
					res->add_dependency(*src_it, dst_action);
			}
		}

		res->drop_actions_not_required_to_do(target_action_set);

		for(std::set<std::string>::const_iterator it = layer_weights_to_update.begin(); it != layer_weights_to_update.end(); ++it)
		{
			layer_action action(layer_action::update_weights);
			std::vector<layer_name_with_action> dependencies;
			dependencies.push_back(layer_name_with_action(*it, layer_action(layer_action::backward_weights)));
			layer::const_ptr l = get_layer(*it);
			for(unsigned int backprop_index = 0; backprop_index < static_cast<unsigned int>(l->input_layer_instance_names.size()); ++backprop_index)
			{
				layer_name_with_action backprop_action(*it, layer_action(layer_action::backward_data, backprop_index));
				if (res->action_exists(backprop_action))
					dependencies.push_back(backprop_action);
			}
			res->add_action(
				l,
				action,
				dependencies);
		}

		for(int i = static_cast<int>(same_output_action_sets.size()) - 1; i >= 0; --i)
		{
			std::vector<layer_name_with_action>& tt = same_output_action_sets[i];
			for(int j = static_cast<int>(tt.size()) - 1; j >= 0; --j)
				if (!res->action_exists(tt[j]))
					tt.erase(tt.begin() + j);

			if (tt.empty())
				same_output_action_sets.erase(same_output_action_sets.begin() + i);
		}

		return res;
	}

	std::vector<layer::const_ptr> network_schema::get_layers() const
	{
		std::vector<layer::const_ptr> res;
		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			res.push_back(layers[*vp.first].l);

		return res;
	}

	std::vector<layer::const_ptr> network_schema::get_data_layers() const
	{
		std::vector<layer::const_ptr> res;
		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			if (layers[*vp.first].l->get_type_name() == data_layer::layer_type_name)
				res.push_back(layers[*vp.first].l);

		return res;
	}

	std::vector<layer::const_ptr> network_schema::get_layers_in_forward_propagation_order() const
	{
		std::list<schema_graph::vertex_descriptor> vertex_list;
		boost::topological_sort(layers, std::back_inserter(vertex_list));
		std::vector<layer::const_ptr> res;
		for(std::list<schema_graph::vertex_descriptor>::const_iterator it = vertex_list.begin(); it != vertex_list.end(); ++it)
			res.push_back(layers[*it].l);
		return res;
	}

	std::map<std::string, layer_configuration_specific> network_schema::get_layer_configuration_specific_map(
		const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map) const
	{
		std::map<std::string, layer_configuration_specific> res(input_configuration_specific_map);

		std::vector<layer::const_ptr> layers_ordered = get_layers_in_forward_propagation_order();
		for(std::vector<layer::const_ptr>::const_iterator it = layers_ordered.begin(); it != layers_ordered.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (res.find(l->instance_name) == res.end())
			{
				std::vector<layer_configuration_specific> input_configuration_specific_list;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
				{
					const std::string& input_layer_instance_name = *it2;
					std::map<std::string, layer_configuration_specific>::const_iterator it_inp = res.find(input_layer_instance_name);
					if (it_inp == res.end())
						throw neural_network_exception((boost::format("Config for layer %1% not found when calculating config for %2%") % input_layer_instance_name % l->instance_name).str());
					input_configuration_specific_list.push_back(it_inp->second);
				}
				res.insert(std::make_pair(l->instance_name, l->get_output_layer_configuration_specific(input_configuration_specific_list)));
			}
		}

		return res;
	}

	std::map<std::string, unsigned int> network_schema::get_cumulative_tiling_factor_map() const
	{
		std::map<std::string, unsigned int> res;

		std::vector<layer::const_ptr> layers_ordered = get_layers_in_forward_propagation_order();
		for(std::vector<layer::const_ptr>::const_iterator it = layers_ordered.begin(); it != layers_ordered.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (l->input_layer_instance_names.empty())
				res.insert(std::make_pair(l->instance_name, 1));
			else
			{
				int current_tiling_factor = -1;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
				{
					const std::string& input_layer_instance_name = *it2;
					std::map<std::string, unsigned int>::const_iterator it_inp = res.find(input_layer_instance_name);
					if (it_inp == res.end())
						throw neural_network_exception((boost::format("Tiling factor for layer %1% not found when calculating factor for %2%") % input_layer_instance_name % l->instance_name).str());
					if (current_tiling_factor == -1)
						current_tiling_factor = static_cast<int>(it_inp->second);
					else
						if (static_cast<unsigned int>(current_tiling_factor) != it_inp->second)
							throw neural_network_exception((boost::format("Tiling factors mismatch for inputs of layer %1%: %2% and %3%") % l->instance_name % current_tiling_factor % it_inp->second).str());
				}
				tiling_factor new_tf = l->get_tiling_factor();
				current_tiling_factor = tiling_factor(current_tiling_factor) * new_tf;
				res.insert(std::make_pair(l->instance_name, current_tiling_factor));			
			}
		}

		return res;
	}

	layer::const_ptr network_schema::find_layer(const std::string& instance_name) const
	{
		std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it = layer_instance_name_to_vertex_decriptor_map.find(instance_name);
		if (it == layer_instance_name_to_vertex_decriptor_map.end())
			return layer::const_ptr();

		return layers[it->second].l;
	}

	layer::const_ptr network_schema::get_layer(const std::string& instance_name) const
	{
		layer::const_ptr res = find_layer(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("Layer not found: %1%") % instance_name).str());

		return res;
	}

	network_schema::gv_vertex_writer::gv_vertex_writer(
		const schema_graph& g,
		const std::map<std::string, unsigned int>& layer_name_color_map)
		: g(g)
		, layer_name_color_map(layer_name_color_map)
	{
	}

	void network_schema::gv_vertex_writer::operator()(std::ostream& out, const network_schema::schema_graph::vertex_descriptor& v) const
	{
		out << " [";
		out << " label=\"" << g[v].l->instance_name << "\"";
		out << " shape=" << ((g[v].l->get_type_name() == data_layer::layer_type_name) ? "parallelogram" : "box") << "";
		
		std::map<std::string, unsigned int>::const_iterator color_it = layer_name_color_map.find(g[v].l->instance_name);
		if (color_it != layer_name_color_map.end())
			out << " style=filled fillcolor=\"" << single_color_palette::get_const_instance().get_color_name(color_it->second) << "\"";

		out << " ]";
	}

	network_schema::gv_graph_writer::gv_graph_writer(const schema_graph& g)
		: g(g)
	{
	}

	void network_schema::gv_graph_writer::operator()(std::ostream& out) const
	{
	}

	void network_schema::write_gv(
		std::ostream& stream_to_write_to,
		const std::map<std::string, unsigned int>& layer_name_color_map) const
	{
		boost::write_graphviz(
			stream_to_write_to, boost::make_reverse_graph(layers),
			gv_vertex_writer(layers, layer_name_color_map),
			boost::default_writer(),
			gv_graph_writer(layers));
	}

	std::map<std::string, layer_configuration_specific> network_schema::get_layer_configuration_specific_map_reverse(const std::map<std::string, layer_configuration_specific>& output_configuration_specific_map) const
	{
		std::map<std::string, layer_configuration_specific> res(output_configuration_specific_map);

		std::vector<layer::const_ptr> layers_ordered = get_layers_in_forward_propagation_order();
		for(std::vector<layer::const_ptr>::const_reverse_iterator it = layers_ordered.rbegin(); it != layers_ordered.rend(); ++it)
		{
			std::map<std::string, layer_configuration_specific>::const_iterator current_config_it = res.find((*it)->instance_name);
			if (current_config_it == res.end())
				continue;
			const layer_configuration_specific& current_config = current_config_it->second;

			layer::const_ptr current_l = *it;
			unsigned int input_layer_id = 0;
			for(std::vector<std::string>::const_iterator it2 = current_l->input_layer_instance_names.begin(); it2 != current_l->input_layer_instance_names.end(); ++it2, ++input_layer_id)
			{
				const std::string& previous_layer_name = *it2;
				if (res.find(previous_layer_name) == res.end())
				{
					layer_configuration_specific input_config;
					if (current_l->get_input_layer_configuration_specific(input_config, current_config, input_layer_id))
						res.insert(std::make_pair(previous_layer_name, input_config));
				}
			}
		}

		return res;
	}
	
/*
	layer_configuration_specific_list network_schema::get_layer_configuration_specific_list_reverse(const layer_configuration_specific& output_layer_configuration_specific) const
	{
		layer_configuration_specific_list res;

		res.resize(layers.size() + 1);
		res[layers.size()] = output_layer_configuration_specific;

		for(unsigned int i = static_cast<unsigned int>(layers.size()); i > 0; --i)
			res[i - 1] = (layers[i - 1]->get_input_layer_configuration_specific(res[i]));

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > network_schema::get_input_rectangle_borders(
		const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		unsigned int output_layer_id) const
	{
		std::vector<std::pair<unsigned int, unsigned int> > input_rectangle_borders = output_rectangle_borders;

		for(int i = static_cast<int>(output_layer_id); i >= 0; --i)
			input_rectangle_borders = layers[i]->get_input_rectangle_borders(input_rectangle_borders);

		return input_rectangle_borders;
	}

	std::vector<unsigned int> network_schema::get_output_strides() const
	{
		std::vector<tiling_factor> min_factors;
		std::vector<tiling_factor> current_factors;
		for(int i = static_cast<int>(layers.size() - 1); i >= 0; --i)
		{
			std::vector<tiling_factor> tf_list = layers[i]->get_tiling_factor_list();

			for(int j = 0; j < tf_list.size(); ++j)
			{
				if (j < current_factors.size())
					current_factors[j] *= tf_list[j];
				else
					current_factors.push_back(tf_list[j]) ;
			}

			for(int j = 0; j < current_factors.size(); ++j)
			{
				if (j < min_factors.size())
					min_factors[j] = std::min(min_factors[j], current_factors[j]);
				else
					min_factors.push_back(current_factors[j]);
			}
		}

		std::vector<unsigned int> res;
		for(int i = 0; i < min_factors.size(); ++i)
		{
			if (min_factors[i] > tiling_factor(1))
				throw neural_network_exception((boost::format("network_schema::get_output_strides - invalid minimum stride %1% encountered at dimension %2%") % min_factors[i].str() % i).str());

			res.push_back(min_factors[i].get_inverse());
		}

		return res;
	}
	*/
}

