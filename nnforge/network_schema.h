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

#pragma once

#include "layer.h"
#include "layer_configuration.h"
#include "layer_configuration_specific.h"
#include "layer_data_configuration.h"
#include "nn_types.h"

#include <vector>
#include <ostream>
#include <istream>
#include <string>
#include <map>
#include <set>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

namespace nnforge
{
	class network_schema
	{
	public:
		typedef nnforge_shared_ptr<network_schema> ptr;
		typedef nnforge_shared_ptr<const network_schema> const_ptr;

		network_schema();

		network_schema(const std::vector<layer::const_ptr>& layer_list);

		void write_proto(std::ostream& stream_to_write_to) const;

		void read_proto(std::istream& stream_to_read_from);

		// The function returns the list of layers required to populate data for the output layers specified
		std::vector<layer::const_ptr> get_required_layers(const std::vector<std::string>& output_layer_names) const;

		// The function returns all the layers
		std::vector<layer::const_ptr> get_layers() const;

		// The functiona returns data layers, that is layers dependent on external data
		std::vector<layer::const_ptr> get_data_layers() const;

		// The function returns all the layers with input layers always preceding corresponding output layers
		// Data layers are included
		std::vector<layer::const_ptr> get_layers_in_forward_propagation_order() const;

		// The function returns sets of layers, each set corresponds to one stream
		// Data layers are not included
		std::vector<std::vector<layer::const_ptr> > get_layer_stream_set_for_forward_propagation() const;

		// The function returns sets of layers, layers in the same set share the same output buffer
		// input_index_layer_can_write_output_map contains info on whether the layer is able to write the output to one of its input
		// separate_buffers_layer_names contains layers for which separate buffers will be allocated
		std::vector<std::vector<layer::const_ptr> > get_layer_buffer_set_for_forward_propagation(
			const std::map<std::string, unsigned int>& input_index_layer_can_write_output_map,
			const std::set<std::string>& separate_buffers_layer_names) const;

		// The function returns sets of layers, layers in the same set share the same working buffer
		// buffers_layer_names defines what layers need working buffer
		std::vector<std::vector<layer::const_ptr> > get_temporary_working_buffer_set_for_forward_propagation(
			const std::set<std::string>& buffers_layer_names) const;

		// The result includes input configurations
		std::map<std::string, layer_configuration_specific> get_layer_configuration_specific_map(const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map) const;

		// The result includes output configuration
		//layer_configuration_specific_list get_layer_configuration_specific_list_reverse(const layer_configuration_specific& output_layer_configuration_specific) const;

		// Returns minimal input rectangle which is quasi-transformed into output one covering the rectangle supplied
		//std::vector<std::pair<unsigned int, unsigned int> > get_input_rectangle_borders(
		//	const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		//	unsigned int output_layer_id) const;

		//std::vector<layer_data_configuration_list> get_layer_data_configuration_list_list() const;

		std::map<std::string, unsigned int> get_cumulative_tiling_factor_map() const;

		//std::vector<unsigned int> get_output_strides() const;

		// Returns empty smart pointer in case layer is not found
		layer::const_ptr find_layer(const std::string& instance_name) const;

		// Throws exception in case layer is not found
		layer::const_ptr get_layer(const std::string& instance_name) const;

		void write_dot(
			std::ostream& stream_to_write_to,
			const std::map<std::string, unsigned int>& layer_name_color_map = std::map<std::string, unsigned int>()) const;

	public:
		std::string name;

	private:
		void clear();

		void add_layer(layer::const_ptr new_layer);

		void add_edges();

		void fill_name_map();

		void detect_cycles();

	private:
		struct vertex_info
		{
			layer::const_ptr l;
		};

		typedef boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::bidirectionalS,
			vertex_info> schema_graph;

		typedef boost::adjacency_list<
			boost::vecS,
			boost::vecS,
			boost::undirectedS,
			vertex_info> undirected_schema_graph;

		schema_graph layers;
		std::map<std::string, schema_graph::vertex_descriptor> layer_instance_name_to_vertex_decriptor_map;

	private:
		static int get_graph_coloring(
			const undirected_schema_graph& graph,
			boost::vector_property_map<undirected_schema_graph::vertex_descriptor>& color);

	private:
		struct cycle_detector : public boost::default_dfs_visitor
		{
			cycle_detector(bool& has_cycle) 
				: has_cycle(has_cycle)
			{
			}

			template <class edge, class graph>
			void back_edge(const edge&, const graph&)
			{
				has_cycle = true;
			}

		protected:
			bool& has_cycle;
		};

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

		struct dot_vertex_writer
		{
			dot_vertex_writer(
				const schema_graph& g,
				const std::map<std::string, unsigned int>& layer_name_color_map);

			void operator()(std::ostream& out, const schema_graph::vertex_descriptor& v) const;

		protected:
			const schema_graph& g;
			const std::map<std::string, unsigned int>& layer_name_color_map;
		};

		struct dot_graph_writer
		{
			dot_graph_writer(const schema_graph& g);

			void operator()(std::ostream& out) const;

		protected:
			const schema_graph& g;
		};

		private:
			static const char * node_color_scheme;
	};
}
