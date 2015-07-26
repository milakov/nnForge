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
	const char * network_schema::node_color_scheme = "set312";

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

	std::vector<std::vector<layer::const_ptr> > network_schema::get_layer_buffer_set_for_forward_propagation(
		const std::map<std::string, unsigned int>& input_index_layer_can_write_output_map,
		const std::set<std::string>& separate_buffers_layer_names) const
	{
		undirected_schema_graph incompatible_output_layers;
		std::map<std::string, undirected_schema_graph::vertex_descriptor> incompatible_output_layer_instance_name_to_vertex_decriptor_map;
		{
			for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			{
				if (separate_buffers_layer_names.find(layers[*vp.first].l->instance_name) == separate_buffers_layer_names.end())
				{
					undirected_schema_graph::vertex_descriptor new_layer_descriptor = boost::add_vertex(incompatible_output_layers);
					incompatible_output_layers[new_layer_descriptor].l = layers[*vp.first].l;
				}
			}
			for(std::pair<undirected_schema_graph::vertex_iterator, undirected_schema_graph::vertex_iterator> vp = boost::vertices(incompatible_output_layers); vp.first != vp.second; ++vp.first)
			{
				undirected_schema_graph::vertex_descriptor layer_descriptor = *vp.first;
				incompatible_output_layer_instance_name_to_vertex_decriptor_map.insert(std::make_pair(incompatible_output_layers[layer_descriptor].l->instance_name, layer_descriptor));
			}
		} // incompatible_output_layers_graph is filled with layers, no edges yet

		std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> > prior_vertices_map;
		{
			for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			{
				schema_graph::vertex_descriptor current_vertex = *vp.first;

				if (separate_buffers_layer_names.find(layers[current_vertex].l->instance_name) != separate_buffers_layer_names.end())
					continue;

				std::set<schema_graph::vertex_descriptor>& prior_vertices = prior_vertices_map.insert(std::make_pair(current_vertex, std::set<schema_graph::vertex_descriptor>())).first->second;
				record_all_edges<schema_graph::vertex_descriptor> vis(prior_vertices);
				std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));
				boost::depth_first_visit(
					layers,
					*vp.first,
					vis,
					boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));
			}
		} // prior_vertices_map is filled with all prior vertices for each vertex

		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
		{
			schema_graph::vertex_descriptor vertex1 = *vp.first;
			std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> >::const_iterator it = prior_vertices_map.find(vertex1);
			if (it != prior_vertices_map.end())
			{
				const std::set<schema_graph::vertex_descriptor>& prior_vertices1 = it->second;
				for(schema_graph::vertex_iterator it2 = vp.first + 1; it2 != vp.second; ++it2)
				{
					schema_graph::vertex_descriptor vertex2 = *it2;
					std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> >::const_iterator it3 = prior_vertices_map.find(vertex2);
					if (it3 != prior_vertices_map.end())
					{
						const std::set<schema_graph::vertex_descriptor>& prior_vertices2 = it3->second;
						bool v1_follows_v2 = (prior_vertices1.find(vertex2) != prior_vertices1.end());
						bool v2_follows_v1 = (prior_vertices2.find(vertex1) != prior_vertices2.end());
						bool incompatible_output_detected = false;
						if (v1_follows_v2 && v2_follows_v1)
							throw neural_network_exception("Cyclic dependency encountered in get_buffer_layer_set_for_forward_propagation");
						else if (!v1_follows_v2 && !v2_follows_v1)
							incompatible_output_detected = true;
						else
						{
							schema_graph::vertex_descriptor prior_vertex = v1_follows_v2 ? vertex2 : vertex1;
							schema_graph::vertex_descriptor next_vertex = v1_follows_v2 ? vertex1 : vertex2;
							const std::set<schema_graph::vertex_descriptor>& prior_vertices = v1_follows_v2 ? prior_vertices1 : prior_vertices2;

							for(std::pair<schema_graph::in_edge_iterator, schema_graph::in_edge_iterator> ep = boost::in_edges(prior_vertex, layers); ep.first != ep.second; ep.first++)
							{
								schema_graph::vertex_descriptor immediate_next_for_prior_vertex = boost::source(*ep.first, layers);
								if (prior_vertices.find(immediate_next_for_prior_vertex) == prior_vertices.end())
								{
									// prior vertex might be yet needed by the time we reach next_vertex
									incompatible_output_detected = true;
									break;
								}

								if (immediate_next_for_prior_vertex == next_vertex)
								{
									// Check, maybe next_vertex is able to write directly to its immediate input prior_vertex
									int input_index_layer_can_write_output = -1;
									std::map<std::string, unsigned int>::const_iterator it_input_index_layer_can_write_output = input_index_layer_can_write_output_map.find(layers[next_vertex].l->instance_name);
									if (it_input_index_layer_can_write_output != input_index_layer_can_write_output_map.end())
										input_index_layer_can_write_output = static_cast<unsigned int>(it_input_index_layer_can_write_output->second);
									if (
										(input_index_layer_can_write_output == -1) ||
										(layers[next_vertex].l->input_layer_instance_names[input_index_layer_can_write_output] != layers[prior_vertex].l->instance_name))
									{
										incompatible_output_detected = true;
										break;
									}
								}
							}
						}

						if (incompatible_output_detected)
						{
							boost::add_edge(
								incompatible_output_layer_instance_name_to_vertex_decriptor_map[layers[vertex1].l->instance_name],
								incompatible_output_layer_instance_name_to_vertex_decriptor_map[layers[vertex2].l->instance_name],
								incompatible_output_layers);
						}
					}
				}
			}
		} // incompatible_output_layers is filled with edges

		boost::vector_property_map<undirected_schema_graph::vertex_descriptor> color;
		//std::vector<vertices_size_type> color_vec(boost::num_vertices(graph));
		//boost::iterator_property_map<vertices_size_type*, vertex_index_map> color(&color_vec.front(), boost::get(boost::vertex_index, graph));
		int color_count = get_graph_coloring(incompatible_output_layers, color);

		std::vector<std::vector<layer::const_ptr> > res(color_count);
		for(std::pair<undirected_schema_graph::vertex_iterator, undirected_schema_graph::vertex_iterator> vp = boost::vertices(incompatible_output_layers); vp.first != vp.second; ++vp.first)
			res[color[*vp.first]].push_back(incompatible_output_layers[*vp.first].l);

		return res;
	}

	std::vector<std::vector<layer::const_ptr> > network_schema::get_temporary_working_buffer_set_for_forward_propagation(
		const std::set<std::string>& buffers_layer_names) const
	{
		undirected_schema_graph incompatible_output_layers;
		std::map<std::string, undirected_schema_graph::vertex_descriptor> incompatible_output_layer_instance_name_to_vertex_decriptor_map;
		{
			for(std::set<std::string>::const_iterator it = buffers_layer_names.begin(); it != buffers_layer_names.end(); ++it)
			{
				undirected_schema_graph::vertex_descriptor new_layer_descriptor = boost::add_vertex(incompatible_output_layers);
				incompatible_output_layers[new_layer_descriptor].l = layers[layer_instance_name_to_vertex_decriptor_map.find(*it)->second].l;
			}
			for(std::pair<undirected_schema_graph::vertex_iterator, undirected_schema_graph::vertex_iterator> vp = boost::vertices(incompatible_output_layers); vp.first != vp.second; ++vp.first)
			{
				undirected_schema_graph::vertex_descriptor layer_descriptor = *vp.first;
				incompatible_output_layer_instance_name_to_vertex_decriptor_map.insert(std::make_pair(incompatible_output_layers[layer_descriptor].l->instance_name, layer_descriptor));
			}
		} // incompatible_output_layers_graph is filled with layers, no edges yet

		std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> > prior_vertices_map;
		{
			for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
			{
				schema_graph::vertex_descriptor current_vertex = *vp.first;

				if (buffers_layer_names.find(layers[current_vertex].l->instance_name) == buffers_layer_names.end())
					continue;

				std::set<schema_graph::vertex_descriptor>& prior_vertices = prior_vertices_map.insert(std::make_pair(current_vertex, std::set<schema_graph::vertex_descriptor>())).first->second;
				record_all_edges<schema_graph::vertex_descriptor> vis(prior_vertices);
				std::vector<boost::default_color_type> color_map(boost::num_vertices(layers));
				boost::depth_first_visit(
					layers,
					*vp.first,
					vis,
					boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, layers)));
			}
		} // prior_vertices_map is filled with all prior vertices for each vertex

		for(std::pair<schema_graph::vertex_iterator, schema_graph::vertex_iterator> vp = boost::vertices(layers); vp.first != vp.second; ++vp.first)
		{
			schema_graph::vertex_descriptor vertex1 = *vp.first;
			std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> >::const_iterator it = prior_vertices_map.find(vertex1);
			if (it != prior_vertices_map.end())
			{
				const std::set<schema_graph::vertex_descriptor>& prior_vertices1 = it->second;
				for(schema_graph::vertex_iterator it2 = vp.first + 1; it2 != vp.second; ++it2)
				{
					schema_graph::vertex_descriptor vertex2 = *it2;
					std::map<schema_graph::vertex_descriptor, std::set<schema_graph::vertex_descriptor> >::const_iterator it3 = prior_vertices_map.find(vertex2);
					if (it3 != prior_vertices_map.end())
					{
						const std::set<schema_graph::vertex_descriptor>& prior_vertices2 = it3->second;
						bool v1_follows_v2 = (prior_vertices1.find(vertex2) != prior_vertices1.end());
						bool v2_follows_v1 = (prior_vertices2.find(vertex1) != prior_vertices2.end());
						if (v1_follows_v2 && v2_follows_v1)
							throw neural_network_exception("Cyclic dependency encountered in get_temporary_working_buffer_set_for_forward_propagation");
						else if (!v1_follows_v2 && !v2_follows_v1)
						{
							boost::add_edge(
								incompatible_output_layer_instance_name_to_vertex_decriptor_map.find(layers[vertex1].l->instance_name)->second,
								incompatible_output_layer_instance_name_to_vertex_decriptor_map.find(layers[vertex2].l->instance_name)->second,
								incompatible_output_layers);
						}
					}
				}
			}
		} // incompatible_output_layers is filled with edges

		boost::vector_property_map<undirected_schema_graph::vertex_descriptor> color;
		//std::vector<vertices_size_type> color_vec(boost::num_vertices(graph));
		//boost::iterator_property_map<vertices_size_type*, vertex_index_map> color(&color_vec.front(), boost::get(boost::vertex_index, graph));
		int color_count = get_graph_coloring(incompatible_output_layers, color);

		std::vector<std::vector<layer::const_ptr> > res(color_count);
		for(std::pair<undirected_schema_graph::vertex_iterator, undirected_schema_graph::vertex_iterator> vp = boost::vertices(incompatible_output_layers); vp.first != vp.second; ++vp.first)
			res[color[*vp.first]].push_back(incompatible_output_layers[*vp.first].l);

		return res;
	}

	int network_schema::get_graph_coloring(
		const undirected_schema_graph& graph,
		boost::vector_property_map<undirected_schema_graph::vertex_descriptor>& color)
	{
		typedef boost::graph_traits<undirected_schema_graph>::vertices_size_type vertices_size_type;
		typedef boost::property_map<undirected_schema_graph, boost::vertex_index_t>::const_type vertex_index_map;

		boost::vector_property_map<undirected_schema_graph::vertex_descriptor> order;
		boost::smallest_last_vertex_ordering(graph, order);

		vertices_size_type num_colors = boost::sequential_vertex_coloring(graph, order, color);
		return static_cast<int>(num_colors);
	}

	std::vector<std::vector<layer::const_ptr> > network_schema::get_layer_stream_set_for_forward_propagation() const
	{
		std::vector<layer::const_ptr> layers_in_forward_propagation_order = get_layers_in_forward_propagation_order();
		std::vector<std::vector<layer::const_ptr> > res;

		std::set<std::string> covered_layer_names;
		for(std::vector<layer::const_ptr>::const_reverse_iterator it = layers_in_forward_propagation_order.rbegin(); it != layers_in_forward_propagation_order.rend(); ++it)
		{
			layer::const_ptr current_layer = *it;
			if (current_layer->get_type_name() == data_layer::layer_type_name)
				continue;
			if (covered_layer_names.find(current_layer->instance_name) != covered_layer_names.end())
				continue;

			res.push_back(std::vector<layer::const_ptr>());
			std::vector<layer::const_ptr>& layers_in_set = res.back();

			while (current_layer)
			{
				layers_in_set.push_back(current_layer);
				covered_layer_names.insert(current_layer->instance_name);

				const std::vector<std::string>& input_layer_instance_names = current_layer->input_layer_instance_names;
				current_layer.reset();
				for(std::vector<std::string>::const_iterator it2 = input_layer_instance_names.begin(); it2 != input_layer_instance_names.end(); ++it2)
				{
					const std::string& new_layer_name = *it2;
					if (covered_layer_names.find(new_layer_name) != covered_layer_names.end())
						continue;
					layer::const_ptr new_layer = layers[layer_instance_name_to_vertex_decriptor_map.find(new_layer_name)->second].l;
					if (new_layer->get_type_name() == data_layer::layer_type_name)
						continue;
					current_layer = new_layer;
					break;
				}
			}
		}

		return res;
	}

	layer::const_ptr network_schema::find_layer(const std::string& instance_name) const
	{
		layer::const_ptr res;

		std::map<std::string, schema_graph::vertex_descriptor>::const_iterator it = layer_instance_name_to_vertex_decriptor_map.find(instance_name);
		if (it == layer_instance_name_to_vertex_decriptor_map.end())
			return res;

		return layers[it->second].l;
	}

	layer::const_ptr network_schema::get_layer(const std::string& instance_name) const
	{
		layer::const_ptr res = find_layer(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("Layer not found: %1%") % instance_name).str());

		return res;
	}

	network_schema::dot_vertex_writer::dot_vertex_writer(
		const schema_graph& g,
		const std::map<std::string, unsigned int>& layer_name_color_map)
		: g(g)
		, layer_name_color_map(layer_name_color_map)
	{
	}

	void network_schema::dot_vertex_writer::operator()(std::ostream& out, const network_schema::schema_graph::vertex_descriptor& v) const
	{
		out << "[";
		out << " label=\"" << g[v].l->instance_name << "\"";
		out << " shape=" << ((g[v].l->get_type_name() == data_layer::layer_type_name) ? "parallelogram" : "box") << "";
		
		std::map<std::string, unsigned int>::const_iterator color_it = layer_name_color_map.find(g[v].l->instance_name);
		if (color_it != layer_name_color_map.end())
			out << " style=filled fillcolor=" << (color_it->second + 1);

		out << " ]";
	}

	network_schema::dot_graph_writer::dot_graph_writer(const schema_graph& g)
		: g(g)
	{
	}

	void network_schema::dot_graph_writer::operator()(std::ostream& out) const
	{
		out << "node [colorscheme=" << node_color_scheme << "]" << std::endl;
	}

	void network_schema::write_dot(
		std::ostream& stream_to_write_to,
		const std::map<std::string, unsigned int>& layer_name_color_map) const
	{
		boost::write_graphviz(
			stream_to_write_to, boost::make_reverse_graph(layers),
			dot_vertex_writer(layers, layer_name_color_map),
			boost::default_writer(),
			dot_graph_writer(layers));
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

