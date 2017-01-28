/*
 *  Copyright 2011-2017 Maxim Milakov
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

#pragma once

#include "layer.h"
#include "layer_name_with_action.h"
#include "layer_action.h"
#include "layer_name_with_action.h"
#include "layer_configuration_specific.h"
#include "buffer_lifetime.h"

#include <vector>
#include <map>
#include <limits>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/reverse_graph.hpp>

namespace nnforge
{
	class network_action_schema
	{
	public:
		typedef std::shared_ptr<network_action_schema> ptr;
		typedef std::shared_ptr<const network_action_schema> const_ptr;

		network_action_schema() = default;

		network_action_schema(const network_action_schema& other);

		void write_gv(
			std::ostream& stream_to_write_to,
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_color_map = std::map<layer_name_with_action, unsigned int>(),
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_border_color_map = std::map<layer_name_with_action, unsigned int>(),
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_bg_color_map = std::map<layer_name_with_action, unsigned int>(),
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_output_edges_color_map = std::map<layer_name_with_action, unsigned int>()) const;

		void add_action(
			layer::const_ptr l,
			layer_action action,
			const std::vector<layer_name_with_action>& dependencies = std::vector<layer_name_with_action>());

		bool action_exists(const layer_name_with_action& layer_and_action) const;

		void add_dependencies(
			const layer_name_with_action& source_layer_and_action,
			const std::vector<layer_name_with_action>& dependencies);

		void add_dependency(
			const layer_name_with_action& source_layer_and_action,
			const layer_name_with_action& destination_layer_and_action);

		bool dependency_exists(
			const layer_name_with_action& source_layer_and_action,
			const layer_name_with_action& destination_layer_and_action) const;

		void add_dependencies_for_distant_otherwise_inependent_actions(
			const std::map<std::string, layer_configuration_specific>& layer_config_map,
			const std::map<std::string, unsigned int>& tiling_factor_map,
			float saturation_flops,
			const std::vector<layer_name_with_action>& action_list_with_no_new_dependencies_to_add_to = std::vector<layer_name_with_action>(),
			const std::vector<layer_name_with_action>& action_list_with_no_new_dependencies_to_add_from = std::vector<layer_name_with_action>());

		float get_flops(
			const std::map<std::string, layer_configuration_specific>& layer_config_map,
			const std::map<std::string, unsigned int>& tiling_factor_map) const;

		std::map<layer_name_with_action, float> get_flops_per_action(
			const std::map<std::string, layer_configuration_specific>& layer_config_map,
			const std::map<std::string, unsigned int>& tiling_factor_map) const;

		std::vector<layer_name_with_action> get_dependencies(const layer_name_with_action& action) const;

		// The function returns all actions in the correct execution order
		std::vector<layer_name_with_action> get_actions_in_execution_order() const;

		// The function returns all actions in the correct execution order
		std::vector<layer_name_with_action> get_actions_in_execution_order(
			const std::map<std::string, layer_configuration_specific>& layer_config_map,
			const std::map<std::string, unsigned int>& tiling_factor_map) const;

		std::vector<layer_name_with_action> get_actions() const;

		// The function returns sets of actions, each set corresponds to one stream
		std::vector<std::vector<layer_name_with_action> > get_action_stream_set(const std::vector<std::vector<layer_name_with_action>>& initial_value = std::vector<std::vector<layer_name_with_action>>()) const;

		// The function returns sets of buffers, buffers in the same set may share the same storage
		// buffers contains all the buffers which should be distributed across sets
		// dependencies lists off buffers each action depends on
		// input_index_layer_can_write_output_map contains info on whether the action is able to write the output to one of its input
		std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > get_buffer_set(
			const std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > >& buffers,
			const std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, bool> > > >& dependencies_and_overwrites,
			const std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >& should_be_placed_into_the_same_buffers) const;

		void drop_actions_not_required_to_do(const std::set<layer_name_with_action>& target_action_set);

		void drop_action_and_reroute_dependencies(const layer_name_with_action& layer_and_action_to_drop);

	private:
		struct vertex_info
		{
			layer_action action;
			layer::const_ptr l;
		};

		typedef boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::bidirectionalS,
			vertex_info> action_schema_graph;

		typedef boost::reverse_graph<action_schema_graph> reverse_action_schema_graph;

		action_schema_graph actions;
		std::map<layer_name_with_action, action_schema_graph::vertex_descriptor> layer_instance_name_with_action_to_vertex_decriptor_map;
		std::map<std::string, layer::const_ptr> name_to_layer_map;

	private:
		action_schema_graph::vertex_descriptor get_vertex_descriptor(const layer_name_with_action& layer_and_action) const;

		std::vector<std::pair<action_schema_graph::vertex_descriptor, std::pair<double, float> > > get_vertex_with_start_and_duration_list(
			const std::map<std::string, layer_configuration_specific>& layer_config_map,
			const std::map<std::string, unsigned int>& tiling_factor_map) const;

	private:
		// Returns empty smart pointer in case layer is not found
		layer::const_ptr find_layer(const std::string& instance_name) const;

		// Throws exception in case layer is not found
		layer::const_ptr get_layer(const std::string& instance_name) const;

	private:

		template<class vertex>
		struct record_all_edges : public boost::default_dfs_visitor
		{
			record_all_edges(std::set<vertex>& visited_vertices) 
				: visited_vertices(visited_vertices)
			{
			}

			template <class graph>
			void discover_vertex(const vertex& v, const graph&)
			{
				visited_vertices.insert(v);
			}

		protected:
			std::set<vertex>& visited_vertices;
		};

		struct gv_vertex_writer
		{
			gv_vertex_writer(
				const reverse_action_schema_graph& g,
				const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_color_map,
				const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_border_color_map,
				const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_bg_color_map);

			void operator()(std::ostream& out, const reverse_action_schema_graph::vertex_descriptor& v) const;

		protected:
			const reverse_action_schema_graph& g;
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_color_map;
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_border_color_map;
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_bg_color_map;
		};

		struct gv_edge_writer
		{
			gv_edge_writer(
				const reverse_action_schema_graph& g,
				const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_output_edges_color_map);

			void operator()(std::ostream& out, const reverse_action_schema_graph::edge_descriptor& v) const;

		protected:
			const reverse_action_schema_graph& g;
			const std::map<layer_name_with_action, unsigned int>& layer_name_with_action_output_edges_color_map;
		};

		struct gv_graph_writer
		{
			gv_graph_writer(const reverse_action_schema_graph& g);

			void operator()(std::ostream& out) const;

		protected:
			const reverse_action_schema_graph& g;
		};

		struct vertex_info_for_buffer_set
		{
			layer::const_ptr l;
			layer_action action;
			buffer_lifetime lifetime;

			vertex_info_for_buffer_set(
				layer::const_ptr l,
				const layer_action& action,
				const buffer_lifetime& lifetime)
				: l(l)
				, action(action)
				, lifetime(lifetime)
			{
			}
		};

		struct vertex_info_list_for_buffer_set
		{
			vertex_info_list_for_buffer_set()
				: buffer_size(-std::numeric_limits<float>::max())
			{
			};

			std::vector<vertex_info_for_buffer_set> buffers;
			float buffer_size;
		};

	private:
		static const unsigned int border_penwidth;
		static const unsigned int arrow_penwidth;
	};
}
