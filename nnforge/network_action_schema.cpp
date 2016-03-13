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

#include "network_action_schema.h"

#include "neural_network_exception.h"
#include "color_palette.h"
#include "min_weight_sequential_vertex_coloring.h"

#include <algorithm>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/smallest_last_ordering.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/copy.hpp>

namespace nnforge
{
	const unsigned int network_action_schema::border_penwidth = 5;
	const unsigned int network_action_schema::arrow_penwidth = 3;

	network_action_schema::network_action_schema()
	{
	}

	network_action_schema::network_action_schema(const network_action_schema& other)
	{
		std::list<action_schema_graph::vertex_descriptor> vertex_list;
		boost::topological_sort(other.actions, std::back_inserter(vertex_list));
		for(std::list<action_schema_graph::vertex_descriptor>::const_iterator it = vertex_list.begin(); it != vertex_list.end(); ++it)
		{
			layer::const_ptr l = other.actions[*it].l;
			layer_action action = other.actions[*it].action;
			std::vector<layer_name_with_action> dependencies;
			for(std::pair<action_schema_graph::out_edge_iterator, action_schema_graph::out_edge_iterator> edge_it = boost::out_edges(*it, other.actions); edge_it.first != edge_it.second; ++edge_it.first)
			{
				action_schema_graph::vertex_descriptor target_vertex = boost::target(*edge_it.first, other.actions);
				dependencies.push_back(layer_name_with_action(other.actions[target_vertex].l->instance_name, other.actions[target_vertex].action));
			}
			add_action(l, action, dependencies);
		}
	}

	bool network_action_schema::action_exists(const layer_name_with_action& layer_and_action) const
	{
		return (layer_instance_name_with_action_to_vertex_decriptor_map.find(layer_and_action) != layer_instance_name_with_action_to_vertex_decriptor_map.end());
	}

	void network_action_schema::add_action(
		layer::const_ptr l,
		layer_action action,
		const std::vector<layer_name_with_action>& dependencies)
	{
		action_schema_graph::vertex_descriptor new_layer_descriptor = boost::add_vertex(actions);
		actions[new_layer_descriptor].action = action;
		actions[new_layer_descriptor].l = l;

		if (!layer_instance_name_with_action_to_vertex_decriptor_map.insert(std::make_pair(layer_name_with_action(l->instance_name, action), new_layer_descriptor)).second)
			throw neural_network_exception((boost::format("There is already %1% action for layer %2%") % action.str() % l->instance_name).str());

		if (!dependencies.empty())
			add_dependencies(layer_name_with_action(l->instance_name, action), dependencies);

		name_to_layer_map.insert(std::make_pair(l->instance_name, l));
	}

	bool network_action_schema::dependency_exists(
		const layer_name_with_action& source_layer_and_action,
		const layer_name_with_action& destination_layer_and_action) const
	{
		action_schema_graph::vertex_descriptor source_vertex = get_vertex_descriptor(source_layer_and_action);
		action_schema_graph::vertex_descriptor destination_vertex = get_vertex_descriptor(destination_layer_and_action);
		return boost::edge(source_vertex, destination_vertex, actions).second;
	}

	void network_action_schema::add_dependencies(
		const layer_name_with_action& source_layer_and_action,
		const std::vector<layer_name_with_action>& dependencies)
	{
		action_schema_graph::vertex_descriptor source_vertex = get_vertex_descriptor(source_layer_and_action);
		for(std::vector<layer_name_with_action>::const_iterator it = dependencies.begin(); it != dependencies.end(); ++it)
		{
			action_schema_graph::vertex_descriptor dest_vertex = get_vertex_descriptor(*it);
			if (!boost::edge(source_vertex, dest_vertex, actions).second)
				boost::add_edge(source_vertex, dest_vertex, actions);
		}
	}

	void network_action_schema::add_dependency(
		const layer_name_with_action& source_layer_and_action,
		const layer_name_with_action& destination_layer_and_action)
	{
		action_schema_graph::vertex_descriptor source_vertex = get_vertex_descriptor(source_layer_and_action);
		action_schema_graph::vertex_descriptor destination_vertex = get_vertex_descriptor(destination_layer_and_action);
		if (!boost::edge(source_vertex, destination_vertex, actions).second)
			boost::add_edge(source_vertex, destination_vertex, actions);
	}

	network_action_schema::action_schema_graph::vertex_descriptor network_action_schema::get_vertex_descriptor(const layer_name_with_action& layer_and_action) const
	{
		std::map<layer_name_with_action, action_schema_graph::vertex_descriptor>::const_iterator it =
			layer_instance_name_with_action_to_vertex_decriptor_map.find(layer_and_action);
		if (it == layer_instance_name_with_action_to_vertex_decriptor_map.end())
			throw neural_network_exception((boost::format("Cannot find %1% action for layer %2%") % layer_and_action.get_action().str() % layer_and_action.get_name()).str());
		return it->second;
	}

	float network_action_schema::get_flops(
		const std::map<std::string, layer_configuration_specific>& layer_config_map,
		const std::map<std::string, unsigned int>& tiling_factor_map) const
	{
		float flops = 0.0F;
		for(std::pair<action_schema_graph::vertex_iterator, action_schema_graph::vertex_iterator> vp = boost::vertices(actions); vp.first != vp.second; ++vp.first)
		{
			layer::const_ptr l = actions[*vp.first].l;
			layer_action action = actions[*vp.first].action;
			std::vector<layer_configuration_specific> input_layer_configs;
			for(std::vector<std::string>::const_iterator it = l->input_layer_instance_names.begin(); it != l->input_layer_instance_names.end(); ++it)
				input_layer_configs.push_back(layer_config_map.find(*it)->second);
			unsigned int tiling_factor = 1;
			std::map<std::string, unsigned int>::const_iterator tiling_it = tiling_factor_map.find(l->instance_name);
			if (tiling_it != tiling_factor_map.end())
				tiling_factor = tiling_it->second;
			flops += l->get_flops_per_entry(input_layer_configs, action) * tiling_factor;
		}
		return flops;
	}

	std::vector<layer_name_with_action> network_action_schema::get_actions_in_execution_order() const
	{
		std::list<action_schema_graph::vertex_descriptor> vertex_list;
		boost::topological_sort(actions, std::back_inserter(vertex_list));
		std::vector<layer_name_with_action> res;
		for(std::list<action_schema_graph::vertex_descriptor>::const_iterator it = vertex_list.begin(); it != vertex_list.end(); ++it)
			res.push_back(layer_name_with_action(actions[*it].l->instance_name, actions[*it].action));
		return res;
	}

	std::vector<layer_name_with_action> network_action_schema::get_actions() const
	{
		std::vector<layer_name_with_action> res;
		action_schema_graph::vertex_iterator vi, vi_end;
		for(boost::tie(vi, vi_end) = boost::vertices(actions); vi != vi_end; ++vi)
			res.push_back(layer_name_with_action(actions[*vi].l->instance_name, actions[*vi].action));
		return res;
	}

	std::vector<layer_name_with_action> network_action_schema::get_dependencies(const layer_name_with_action& action) const
	{
		std::vector<layer_name_with_action> res;
		action_schema_graph::adjacency_iterator vi, vi_end;
		for(boost::tie(vi, vi_end) = boost::adjacent_vertices(layer_instance_name_with_action_to_vertex_decriptor_map.find(action)->second, actions); vi != vi_end; ++vi)
			res.push_back(layer_name_with_action(actions[*vi].l->instance_name, actions[*vi].action));
		return res;
	}

	std::vector<std::vector<layer_name_with_action> > network_action_schema::get_action_stream_set() const
	{
		std::vector<std::vector<layer_name_with_action> > res;

		std::vector<action_schema_graph::vertex_descriptor> vertex_list_in_execution_order;
		boost::topological_sort(actions, std::back_inserter(vertex_list_in_execution_order));

		std::set<layer_name_with_action> covered_layer_name_with_actions;
		for(std::vector<action_schema_graph::vertex_descriptor>::const_reverse_iterator it = vertex_list_in_execution_order.rbegin(); it != vertex_list_in_execution_order.rend(); ++it)
		{
			layer::const_ptr current_layer = actions[*it].l;
			layer_action action = actions[*it].action;
			layer_name_with_action current_layer_name_with_action(current_layer->instance_name, action);
			if (covered_layer_name_with_actions.find(current_layer_name_with_action) != covered_layer_name_with_actions.end())
				continue;

			res.push_back(std::vector<layer_name_with_action>());
			std::vector<layer_name_with_action>& actions_in_set = res.back();

			bool current_set = true;
			while (current_set)
			{
				actions_in_set.push_back(current_layer_name_with_action);
				covered_layer_name_with_actions.insert(current_layer_name_with_action);

				current_set = false;
				action_schema_graph::vertex_descriptor current_vertex = layer_instance_name_with_action_to_vertex_decriptor_map.find(current_layer_name_with_action)->second;
				for(std::pair<action_schema_graph::adjacency_iterator, action_schema_graph::adjacency_iterator> vp = boost::adjacent_vertices(current_vertex, actions); vp.first != vp.second; ++vp.first)
				{
					layer::const_ptr current_layer = actions[*vp.first].l;
					layer_action action = actions[*vp.first].action;
					layer_name_with_action new_layer_name_with_action = layer_name_with_action(current_layer->instance_name, action);
					if (covered_layer_name_with_actions.find(new_layer_name_with_action) == covered_layer_name_with_actions.end())
					{
						current_layer_name_with_action = new_layer_name_with_action;
						current_set = true;
						break;
					}
				}

				// All direct priors are already covered
				if (!current_set)
				{
					std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
					record_all_edges<action_schema_graph::vertex_descriptor> vis(dependent_vertices);
					std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
					boost::depth_first_visit(
						actions,
						current_vertex,
						vis,
						boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));
					for(std::vector<action_schema_graph::vertex_descriptor>::const_reverse_iterator it2 = it + 1; it2 != vertex_list_in_execution_order.rend(); ++it2)
					{
						if (dependent_vertices.find(*it2) != dependent_vertices.end())
						{
							layer::const_ptr current_layer = actions[*it2].l;
							layer_action action = actions[*it2].action;
							layer_name_with_action new_layer_name_with_action = layer_name_with_action(current_layer->instance_name, action);
							if (covered_layer_name_with_actions.find(new_layer_name_with_action) == covered_layer_name_with_actions.end())
							{
								current_layer_name_with_action = layer_name_with_action(current_layer->instance_name, action);
								current_set = true;
								break;
							}
						}
					}
				}
			}
		}

		return res;
	}

	network_action_schema::gv_vertex_writer::gv_vertex_writer(
		const reverse_action_schema_graph& g,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_color_map,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_border_color_map,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_bg_color_map)
		: g(g)
		, layer_name_with_action_color_map(layer_name_with_action_color_map)
		, layer_name_with_action_border_color_map(layer_name_with_action_border_color_map)
		, layer_name_with_action_bg_color_map(layer_name_with_action_bg_color_map)
	{
	}

	void network_action_schema::gv_vertex_writer::operator()(std::ostream& out, const network_action_schema::reverse_action_schema_graph::vertex_descriptor& v) const
	{
		out << " [";
		out << " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD><B>" << g[v].l->instance_name << "</B></TD></TR><TR><TD";

		std::map<layer_name_with_action, unsigned int>::const_iterator bg_color_it = layer_name_with_action_bg_color_map.find(layer_name_with_action(g[v].l->instance_name, g[v].action));
		if (bg_color_it != layer_name_with_action_bg_color_map.end())
			out << " BGCOLOR=\"" << single_color_palette::get_const_instance().get_color_name(bg_color_it->second) << "\"";

		out << ">" << g[v].action.str() << "</TD></TR></TABLE>>";
		out << " shape=rect";
		
		std::map<layer_name_with_action, unsigned int>::const_iterator color_it = layer_name_with_action_color_map.find(layer_name_with_action(g[v].l->instance_name, g[v].action));
		if (color_it != layer_name_with_action_color_map.end())
			out << " style=filled fillcolor=\"" << single_color_palette::get_const_instance().get_color_name(color_it->second) << "\"";

		std::map<layer_name_with_action, unsigned int>::const_iterator border_color_it = layer_name_with_action_border_color_map.find(layer_name_with_action(g[v].l->instance_name, g[v].action));
		if (border_color_it != layer_name_with_action_border_color_map.end())
			out << " penwidth=" << border_penwidth << " color=\"" << single_color_palette::get_const_instance().get_color_name(border_color_it->second) << "\"";

		out << " ]";
	}

	network_action_schema::gv_edge_writer::gv_edge_writer(
		const reverse_action_schema_graph& g,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_output_edges_color_map)
		: g(g)
		, layer_name_with_action_output_edges_color_map(layer_name_with_action_output_edges_color_map)
	{
	}

	void network_action_schema::gv_edge_writer::operator()(std::ostream& out, const network_action_schema::reverse_action_schema_graph::edge_descriptor& e) const
	{
		out << " [";

		network_action_schema::reverse_action_schema_graph::vertex_descriptor source_v = boost::source(e, g);
		std::map<layer_name_with_action, unsigned int>::const_iterator output_edge_color_it = layer_name_with_action_output_edges_color_map.find(layer_name_with_action(g[source_v].l->instance_name, g[source_v].action));
		if (output_edge_color_it != layer_name_with_action_output_edges_color_map.end())
			out << " penwidth=" << arrow_penwidth << " color=\"" << single_color_palette::get_const_instance().get_color_name(output_edge_color_it->second) << "\"";

		out << " ]";
	}

	network_action_schema::gv_graph_writer::gv_graph_writer(const reverse_action_schema_graph& g)
		: g(g)
	{
	}

	void network_action_schema::gv_graph_writer::operator()(std::ostream& out) const
	{
	}

	void network_action_schema::write_gv(
		std::ostream& stream_to_write_to,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_color_map,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_border_color_map,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_bg_color_map,
		const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_output_edges_color_map) const
	{
		reverse_action_schema_graph reverse_actions = boost::make_reverse_graph(actions);

		boost::write_graphviz(
			stream_to_write_to,
			boost::make_reverse_graph(actions),
			gv_vertex_writer(actions, layer_name_with_action_color_map, layer_name_with_action_border_color_map, layer_name_with_action_bg_color_map),
			gv_edge_writer(reverse_actions, layer_name_with_action_output_edges_color_map),
			gv_graph_writer(actions));
	}

	template <class VertexListGraph, class ColorMap, class WeightMap>
	int get_graph_coloring(
		const VertexListGraph& graph,
		ColorMap& colors,
		const WeightMap& weights)
	{
		typedef typename boost::graph_traits<VertexListGraph>::vertex_descriptor vertex_descriptor;

		if (boost::num_vertices(graph) == 0)
			return 0;

		std::vector<vertex_descriptor> order(boost::num_vertices(graph));
		boost::smallest_last_vertex_ordering(graph, boost::make_iterator_property_map(order.begin(), boost::identity_property_map(), boost::graph_traits<VertexListGraph>::null_vertex()));
		typename boost::graph_traits<VertexListGraph>::vertices_size_type num_colors = min_weight_sequential_vertex_coloring(graph, boost::make_iterator_property_map(order.begin(), boost::identity_property_map(), boost::graph_traits<VertexListGraph>::null_vertex()), colors, weights);
		return static_cast<int>(num_colors);
	}

	std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > network_action_schema::get_buffer_set(
		const std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >& buffers,
		const std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<buffer_lifetime> > >& dependencies,
		const std::map<layer_name_with_action, unsigned int>& input_index_layer_can_write_output_map,
		const std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >& should_be_placed_into_the_same_buffers) const
	{
		typedef boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::undirectedS,
			vertex_info_list_for_buffer_set> undirected_action_schema_graph2;

		undirected_action_schema_graph2 incompatible_output_actions_with_lifetime;
		std::map<layer_name_with_action, std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor> > incompatible_output_action_to_vertex_decriptor_map;
		{
			for(std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >::const_iterator itt = should_be_placed_into_the_same_buffers.begin(); itt != should_be_placed_into_the_same_buffers.end(); ++itt)
			{
				undirected_action_schema_graph2::vertex_descriptor new_action_descriptor = boost::add_vertex(incompatible_output_actions_with_lifetime);
				const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& buffers = *itt;
				for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = buffers.begin(); it != buffers.end(); ++it)
				{
					layer::const_ptr l = get_layer(it->first.get_name());
					layer_action action = it->first.get_action();
					const buffer_lifetime& lifetime = it->second;

					std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor>& tt = incompatible_output_action_to_vertex_decriptor_map.insert(
						std::make_pair(
							layer_name_with_action(l->instance_name, action),
							std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor>())).first->second;

					if (tt.find(lifetime) != tt.end())
						throw neural_network_exception((boost::format("Buffer %1% for action %2% for layer %3% is specified multiple times for different same buffer sets") % lifetime.str() % l->instance_name % action.str()).str());

					incompatible_output_actions_with_lifetime[new_action_descriptor].buffers.push_back(vertex_info_for_buffer_set(l, action, lifetime));
					tt.insert(std::make_pair(lifetime, new_action_descriptor));
				}
			}

			for(std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >::const_iterator it = buffers.begin(); it != buffers.end(); ++it)
			{
				layer::const_ptr l = get_layer(it->first.get_name());
				layer_action action = it->first.get_action();
				const std::vector<std::pair<buffer_lifetime, float> >& lifetime_list = it->second;

				std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor>& tt = incompatible_output_action_to_vertex_decriptor_map.insert(
					std::make_pair(
						layer_name_with_action(l->instance_name, action),
						std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor>())).first->second;
				for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator it2 = lifetime_list.begin(); it2 != lifetime_list.end(); ++it2)
				{
					const buffer_lifetime& lifetime = it2->first;
					float buffer_size = it2->second;
					std::map<buffer_lifetime, undirected_action_schema_graph2::vertex_descriptor>::iterator old_it = tt.find(lifetime);
					undirected_action_schema_graph2::vertex_descriptor new_action_descriptor;
					if (old_it != tt.end())
					{
						new_action_descriptor = old_it->second;
					}
					else
					{
						new_action_descriptor = boost::add_vertex(incompatible_output_actions_with_lifetime);
						incompatible_output_actions_with_lifetime[new_action_descriptor].buffers.push_back(vertex_info_for_buffer_set(l, action, lifetime));
						tt.insert(std::make_pair(lifetime, new_action_descriptor));
					}

					incompatible_output_actions_with_lifetime[new_action_descriptor].buffer_size = std::max(incompatible_output_actions_with_lifetime[new_action_descriptor].buffer_size, buffer_size);
				}
			}
		} // incompatible_output_actions_with_lifetime is filled with layers, no edges yet

		std::map<layer_name_with_action, std::map<buffer_lifetime, std::set<layer_name_with_action> > > reverse_dependencies;
		{
			for(std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<buffer_lifetime> > >::const_iterator it = dependencies.begin(); it != dependencies.end(); ++it)
			{
				const layer_name_with_action& dependent_action = it->first;
				const std::map<layer_name_with_action, std::vector<buffer_lifetime> >& source_buffers = it->second;
				for(std::map<layer_name_with_action, std::vector<buffer_lifetime> >::const_iterator it2 = source_buffers.begin(); it2 != source_buffers.end(); ++it2)
				{
					const layer_name_with_action& source_action = it2->first;
					const std::vector<buffer_lifetime>& source_buffer_lifetimes = it2->second;

					std::map<buffer_lifetime, std::set<layer_name_with_action> >& t = reverse_dependencies.insert(std::make_pair(source_action, std::map<buffer_lifetime, std::set<layer_name_with_action> >())).first->second;
					for(std::vector<buffer_lifetime>::const_iterator it3 = source_buffer_lifetimes.begin(); it3 != source_buffer_lifetimes.end(); ++it3)
					{
						const buffer_lifetime& source_lifetime = *it3;
						t.insert(std::make_pair(source_lifetime, std::set<layer_name_with_action>())).first->second.insert(dependent_action);
					}
				}
			}
		} // Reverse dependencies are filled

		std::vector<layer_name_with_action> actions_in_execution_order = get_actions_in_execution_order();
		for(std::vector<layer_name_with_action>::const_iterator current_action_it = actions_in_execution_order.begin(); current_action_it != actions_in_execution_order.end(); ++current_action_it)
		{
			const layer_name_with_action& current_layer_name_with_action = *current_action_it;
			std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >::const_iterator current_buffer_list_it = buffers.find(current_layer_name_with_action);
			if (current_buffer_list_it == buffers.end())
				continue;
			const std::vector<std::pair<buffer_lifetime, float> >& current_buffer_lifetimes = current_buffer_list_it->second;

			// Add incompatibility edges for the buffers of the same action
			for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator it1 = current_buffer_lifetimes.begin(); it1 != current_buffer_lifetimes.end(); ++it1)
			{
				for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator it2 = it1 + 1; it2 != current_buffer_lifetimes.end(); ++it2)
				{
					undirected_action_schema_graph2::vertex_descriptor v1 = incompatible_output_action_to_vertex_decriptor_map[current_layer_name_with_action][it1->first];
					undirected_action_schema_graph2::vertex_descriptor v2 = incompatible_output_action_to_vertex_decriptor_map[current_layer_name_with_action][it2->first];
					if ((v1 != v2) && (!boost::edge(v1, v2, incompatible_output_actions_with_lifetime).second))
					{
						boost::add_edge(
							v1,
							v2,
							incompatible_output_actions_with_lifetime);
					}
				}
			}

			std::map<layer_name_with_action, std::map<buffer_lifetime, std::set<layer_name_with_action> > >::const_iterator reverse_dependencies_it = reverse_dependencies.find(current_layer_name_with_action);
			for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator source_buffer_lifetime_it = current_buffer_lifetimes.begin(); source_buffer_lifetime_it != current_buffer_lifetimes.end(); ++source_buffer_lifetime_it)
			{
				const buffer_lifetime& source_buffer_lifetime = source_buffer_lifetime_it->first;

				std::set<layer_name_with_action> dependent_actions;
				if (reverse_dependencies_it != reverse_dependencies.end())
				{
					std::map<buffer_lifetime, std::set<layer_name_with_action> >::const_iterator reverse_dependencies_it2 = reverse_dependencies_it->second.find(source_buffer_lifetime);
					if (reverse_dependencies_it2 != reverse_dependencies_it->second.end())
						dependent_actions = reverse_dependencies_it2->second;
				}

				if (dependent_actions.empty())
				{
					std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
					record_all_edges<action_schema_graph::vertex_descriptor> vis(dependent_vertices);
					std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
					boost::depth_first_visit(
						boost::make_reverse_graph(actions),
						layer_instance_name_with_action_to_vertex_decriptor_map.find(current_layer_name_with_action)->second,
						vis,
						boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));
					for(std::vector<layer_name_with_action>::const_iterator subsequent_action_it = current_action_it + 1; subsequent_action_it != actions_in_execution_order.end(); ++subsequent_action_it)
					{
						const layer_name_with_action& subsequent_layer_name_with_action = *subsequent_action_it;
						std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >::const_iterator subsequent_buffer_list_it = buffers.find(subsequent_layer_name_with_action);
						if (subsequent_buffer_list_it == buffers.end())
							continue;
						if (dependent_vertices.find(layer_instance_name_with_action_to_vertex_decriptor_map.find(subsequent_layer_name_with_action)->second) != dependent_vertices.end())
							continue;
						const std::vector<std::pair<buffer_lifetime, float> >& incompatible_buffer_lifetimes = subsequent_buffer_list_it->second;
						for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator incompatible_buffer_lifetime_it = incompatible_buffer_lifetimes.begin(); incompatible_buffer_lifetime_it != incompatible_buffer_lifetimes.end(); ++incompatible_buffer_lifetime_it)
						{
							undirected_action_schema_graph2::vertex_descriptor v1 = incompatible_output_action_to_vertex_decriptor_map[current_layer_name_with_action][source_buffer_lifetime];
							undirected_action_schema_graph2::vertex_descriptor v2 = incompatible_output_action_to_vertex_decriptor_map[subsequent_layer_name_with_action][incompatible_buffer_lifetime_it->first];
							if ((v1 != v2) && (!boost::edge(v1, v2, incompatible_output_actions_with_lifetime).second))
							{
								boost::add_edge(
									v1,
									v2,
									incompatible_output_actions_with_lifetime);
							}
						}
					}
				}
				else
				{
					std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
					for(std::set<layer_name_with_action>::const_iterator dependent_action_it = dependent_actions.begin(); dependent_action_it != dependent_actions.end(); ++dependent_action_it)
					{
						std::set<action_schema_graph::vertex_descriptor> current_dependent_vertices;
						record_all_edges<action_schema_graph::vertex_descriptor> vis(current_dependent_vertices);
						std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
						boost::depth_first_visit(
							boost::make_reverse_graph(actions),
							layer_instance_name_with_action_to_vertex_decriptor_map.find(*dependent_action_it)->second,
							vis,
							boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));
						if (dependent_action_it == dependent_actions.begin())
							dependent_vertices = current_dependent_vertices;
						else
						{
							std::vector<action_schema_graph::vertex_descriptor> temp(std::min(dependent_vertices.size(), current_dependent_vertices.size()));
							std::vector<action_schema_graph::vertex_descriptor>::iterator temp_it_end = std::set_intersection(dependent_vertices.begin(), dependent_vertices.end(), current_dependent_vertices.begin(), current_dependent_vertices.end(), temp.begin());
							dependent_vertices = std::set<action_schema_graph::vertex_descriptor>(temp.begin(), temp_it_end);
						}
						if (dependent_vertices.empty())
							break; // earlier exit
					}
					for(std::vector<layer_name_with_action>::const_iterator subsequent_action_it = current_action_it + 1; subsequent_action_it != actions_in_execution_order.end(); ++subsequent_action_it)
					{
						const layer_name_with_action& subsequent_layer_name_with_action = *subsequent_action_it;
						std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >::const_iterator subsequent_buffer_list_it = buffers.find(subsequent_layer_name_with_action);
						if (subsequent_buffer_list_it == buffers.end())
							continue;
						if ((dependent_vertices.find(layer_instance_name_with_action_to_vertex_decriptor_map.find(subsequent_layer_name_with_action)->second) != dependent_vertices.end())
							&& (dependent_actions.find(subsequent_layer_name_with_action) == dependent_actions.end()))
							continue; // inside dependency cone
						bool can_overwrite_input = false;
						if ((source_buffer_lifetime.get_buffer_lifetime_type() == buffer_lifetime::action_output_buffer)
							&& (dependent_vertices.find(layer_instance_name_with_action_to_vertex_decriptor_map.find(subsequent_layer_name_with_action)->second) != dependent_vertices.end())
							&& (dependent_actions.find(subsequent_layer_name_with_action) != dependent_actions.end()))
						{
							// should check for border cases
							std::map<layer_name_with_action, unsigned int>::const_iterator input_index_it = input_index_layer_can_write_output_map.find(subsequent_layer_name_with_action);
							if (input_index_it != input_index_layer_can_write_output_map.end())
							{
								switch (subsequent_layer_name_with_action.get_action().get_action_type())
								{
								case layer_action::forward:
									can_overwrite_input = (get_layer(subsequent_layer_name_with_action.get_name())->input_layer_instance_names[input_index_it->second] == current_layer_name_with_action.get_name());
									break;
								case layer_action::backward_data:
									can_overwrite_input = true;
									break;
								case layer_action::backward_data_and_weights:
									can_overwrite_input = true;
									break;
								}
							}
						}
						const std::vector<std::pair<buffer_lifetime, float> >& incompatible_buffer_lifetimes = subsequent_buffer_list_it->second;
						for(std::vector<std::pair<buffer_lifetime, float> >::const_iterator incompatible_buffer_lifetime_it = incompatible_buffer_lifetimes.begin(); incompatible_buffer_lifetime_it != incompatible_buffer_lifetimes.end(); ++incompatible_buffer_lifetime_it)
						{
							if (can_overwrite_input && (incompatible_buffer_lifetime_it->first.get_buffer_lifetime_type() == buffer_lifetime::action_output_buffer))
								continue;

							undirected_action_schema_graph2::vertex_descriptor v1 = incompatible_output_action_to_vertex_decriptor_map[current_layer_name_with_action][source_buffer_lifetime];
							undirected_action_schema_graph2::vertex_descriptor v2 = incompatible_output_action_to_vertex_decriptor_map[subsequent_layer_name_with_action][incompatible_buffer_lifetime_it->first];
							if ((v1 != v2) && (!boost::edge(v1, v2, incompatible_output_actions_with_lifetime).second))
							{
								boost::add_edge(
									v1,
									v2,
									incompatible_output_actions_with_lifetime);
							}
						}
					}
				}
			}
		}
		// incompatible_output_layers is filled with edges

		std::vector<float> weights_vec(boost::num_vertices(incompatible_output_actions_with_lifetime));
		if (weights_vec.empty())
			weights_vec.resize(1); // So that weights_vec.front() would not fail
		boost::iterator_property_map<float*, typename boost::property_map<undirected_action_schema_graph2, boost::vertex_index_t>::const_type> weights(&weights_vec.front(), boost::get(boost::vertex_index, incompatible_output_actions_with_lifetime));
		for(std::pair<undirected_action_schema_graph2::vertex_iterator, undirected_action_schema_graph2::vertex_iterator> vp = boost::vertices(incompatible_output_actions_with_lifetime); vp.first != vp.second; ++vp.first)
			boost::put(weights, *vp.first, incompatible_output_actions_with_lifetime[*vp.first].buffer_size);

		std::vector<typename boost::graph_traits<undirected_action_schema_graph2>::vertices_size_type> colors_vec(boost::num_vertices(incompatible_output_actions_with_lifetime));
		if (colors_vec.empty())
			colors_vec.resize(1); // So that colors_vec.front() would not fail
		boost::iterator_property_map<typename boost::graph_traits<undirected_action_schema_graph2>::vertices_size_type*, typename boost::property_map<undirected_action_schema_graph2, boost::vertex_index_t>::const_type> colors(&colors_vec.front(), boost::get(boost::vertex_index, incompatible_output_actions_with_lifetime));
		int color_count = get_graph_coloring(incompatible_output_actions_with_lifetime, colors, weights);

		std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > res(color_count);
		for(std::pair<undirected_action_schema_graph2::vertex_iterator, undirected_action_schema_graph2::vertex_iterator> vp = boost::vertices(incompatible_output_actions_with_lifetime); vp.first != vp.second; ++vp.first)
			for(std::vector<vertex_info_for_buffer_set>::const_iterator it = incompatible_output_actions_with_lifetime[*vp.first].buffers.begin(); it != incompatible_output_actions_with_lifetime[*vp.first].buffers.end(); ++it)
				res[boost::get(colors, *vp.first)].push_back(std::make_pair(layer_name_with_action(it->l->instance_name, it->action), it->lifetime));

		return res;
	}

	layer::const_ptr network_action_schema::find_layer(const std::string& instance_name) const
	{
		std::map<std::string, layer::const_ptr>::const_iterator it = name_to_layer_map.find(instance_name);
		if (it == name_to_layer_map.end())
			return layer::const_ptr();

		return it->second;
	}

	layer::const_ptr network_action_schema::get_layer(const std::string& instance_name) const
	{
		layer::const_ptr res = find_layer(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("Layer not found: %1%") % instance_name).str());

		return res;
	}

	void network_action_schema::drop_actions_not_required_to_do(const std::set<layer_name_with_action>& target_action_set)
	{
		bool vertex_removed = true;
		while(vertex_removed)
		{
			vertex_removed = false;
			for(std::pair<action_schema_graph::vertex_iterator, action_schema_graph::vertex_iterator> vp = boost::vertices(actions); vp.first != vp.second; ++vp.first)
			{
				if ((boost::in_degree(*vp.first, actions) == 0) && (target_action_set.find(layer_name_with_action(actions[*vp.first].l->instance_name, actions[*vp.first].action)) == target_action_set.end()))
				{
					boost::clear_vertex(*vp.first, actions);
					boost::remove_vertex(*vp.first, actions);
					vertex_removed = true;
					break;
				}
			}
		}

		layer_instance_name_with_action_to_vertex_decriptor_map.clear();
		name_to_layer_map.clear();
		for(std::pair<action_schema_graph::vertex_iterator, action_schema_graph::vertex_iterator> vp = boost::vertices(actions); vp.first != vp.second; ++vp.first)
		{
			layer::const_ptr l = actions[*vp.first].l;
			layer_action action = actions[*vp.first].action;
			name_to_layer_map.insert(std::make_pair(l->instance_name, l));
			layer_instance_name_with_action_to_vertex_decriptor_map.insert(std::make_pair(layer_name_with_action(l->instance_name, action), *vp.first));
		}
	}

	bool network_action_schema::compare_distances(const std::pair<action_schema_graph::vertex_descriptor, double>& t1, const std::pair<action_schema_graph::vertex_descriptor, double>& t2)
	{
		return (t1.second < t2.second);
	}

	std::vector<std::pair<network_action_schema::action_schema_graph::vertex_descriptor, double> > network_action_schema::get_vertex_with_distance_list(
		const std::map<std::string, layer_configuration_specific>& layer_config_map,
		const std::map<std::string, unsigned int>& tiling_factor_map) const
	{
		std::vector<std::pair<action_schema_graph::vertex_descriptor, double> > vertex_distance_list;
		{
			std::map<action_schema_graph::vertex_descriptor, double> distance_map; // Max distance to reach this node
			std::list<action_schema_graph::vertex_descriptor> vertex_list;
			boost::topological_sort(actions, std::back_inserter(vertex_list));
			for(std::list<action_schema_graph::vertex_descriptor>::const_iterator it = vertex_list.begin(); it != vertex_list.end(); ++it)
			{
				double max_distance = 0.0;
				for(std::pair<action_schema_graph::out_edge_iterator, action_schema_graph::out_edge_iterator> edge_it = boost::out_edges(*it, actions); edge_it.first != edge_it.second; ++edge_it.first)
				{
					action_schema_graph::vertex_descriptor previous_vertex = boost::target(*edge_it.first, actions);
					double current_distance = distance_map[previous_vertex];

					float flops;
					{
						layer::const_ptr l = actions[previous_vertex].l;
						std::vector<layer_configuration_specific> input_layer_configs;
						for(std::vector<std::string>::const_iterator it = l->input_layer_instance_names.begin(); it != l->input_layer_instance_names.end(); ++it)
							input_layer_configs.push_back(layer_config_map.find(*it)->second);
						unsigned int tiling_factor = 1;
						std::map<std::string, unsigned int>::const_iterator tiling_it = tiling_factor_map.find(l->instance_name);
						if (tiling_it != tiling_factor_map.end())
							tiling_factor = tiling_it->second;
						flops = l->get_flops_per_entry(input_layer_configs, actions[previous_vertex].action) * tiling_factor;
					}
					current_distance += static_cast<double>(flops);
					max_distance = std::max(max_distance, current_distance);
				}
				distance_map.insert(std::make_pair(*it, max_distance));
				vertex_distance_list.push_back(std::make_pair(*it, max_distance));
			}
			std::stable_sort(vertex_distance_list.begin(), vertex_distance_list.end(), compare_distances);
			// vertex_distance_list is now vertecices in execution order AND with non-decreasing distance to reach each of them
		}

		return vertex_distance_list;
	}

	void network_action_schema::add_dependencies_for_distant_otherwise_inependent_actions(
		const std::map<std::string, layer_configuration_specific>& layer_config_map,
		const std::map<std::string, unsigned int>& tiling_factor_map,
		float saturation_flops)
	{
		std::vector<std::pair<action_schema_graph::vertex_descriptor, double> > vertex_distance_list = get_vertex_with_distance_list(layer_config_map, tiling_factor_map);

		for(std::vector<std::pair<action_schema_graph::vertex_descriptor, double> >::const_iterator it = vertex_distance_list.begin(); it != vertex_distance_list.end(); ++it)
		{
			std::vector<std::pair<action_schema_graph::vertex_descriptor, double> >::const_iterator candidate_it = it + 1;
			double min_distance = it->second + static_cast<double>(saturation_flops);
			while ((candidate_it != vertex_distance_list.end()) && (candidate_it->second < min_distance))
				++candidate_it;

			if (candidate_it == vertex_distance_list.end())
				continue;

			bool edge_addded = true;
			while (edge_addded)
			{
				std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
				record_all_edges<action_schema_graph::vertex_descriptor> vis(dependent_vertices);
				std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
				boost::depth_first_visit(
					boost::make_reverse_graph(actions),
					it->first,
					vis,
					boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));

				edge_addded = false;
				while (candidate_it != vertex_distance_list.end())
				{
					if (dependent_vertices.find(candidate_it->first) == dependent_vertices.end())
					{
						// To reduce the number of dependencies we better remove all the from candidate, where target is prior to current 
						{
							std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
							record_all_edges<action_schema_graph::vertex_descriptor> vis(dependent_vertices);
							std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
							boost::depth_first_visit(
								actions,
								it->first,
								vis,
								boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));
							for(std::set<action_schema_graph::vertex_descriptor>::const_iterator dep_it = dependent_vertices.begin(); dep_it != dependent_vertices.end(); ++dep_it)
							{
								if (boost::edge(candidate_it->first, *dep_it, actions).second)
									boost::remove_edge(candidate_it->first, *dep_it, actions);
							}
						}

						// To reduce the number of dependencies we better remove all the vertices to the target, where source is the descendent to the candidate vertex
						{
							std::set<action_schema_graph::vertex_descriptor> dependent_vertices;
							record_all_edges<action_schema_graph::vertex_descriptor> vis(dependent_vertices);
							std::vector<boost::default_color_type> color_map(boost::num_vertices(actions));
							boost::depth_first_visit(
								boost::make_reverse_graph(actions),
								candidate_it->first,
								vis,
								boost::make_iterator_property_map(color_map.begin(), boost::get(boost::vertex_index, actions)));
							for(std::set<action_schema_graph::vertex_descriptor>::const_iterator dep_it = dependent_vertices.begin(); dep_it != dependent_vertices.end(); ++dep_it)
							{
								if (boost::edge(*dep_it, it->first, actions).second)
									boost::remove_edge(*dep_it, it->first, actions);
							}
						}

						boost::add_edge(candidate_it->first, it->first, actions);
						edge_addded = true;
						break;
					}

					++candidate_it;
				}
			} // while (edge_addded)
		}
	}
}
