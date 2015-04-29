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

#include "network_analyzer.h"

#include "neural_network_exception.h"
#include <boost/format.hpp>

namespace nnforge
{
	network_analyzer::network_analyzer(network_schema_smart_ptr schema)
		: schema(schema)
	{
		const const_layer_list& layers = *schema;
		for(unsigned int i = 0; i < layers.size(); ++i)
		{
			tiling_factor new_tf = layers[i]->get_tiling_factor();
			if (new_tf != 1)
				throw neural_network_exception((boost::format("network_analyzer cannot run layer %1% with non-unit tiling factor") % i).str());
		}
	}

	network_analyzer::~network_analyzer()
	{
	}

	void network_analyzer::set_data(network_data_smart_ptr data)
	{
		// Check data-schema consistency
		data->check_network_data_consistency(*schema);

		actual_set_data(data);
	}

	void network_analyzer::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
	{
		if ((layer_config_list.size() > 0) && (layer_config_list[0] == input_configuration_specific))
			return;

		layer_config_list = schema->get_layer_configuration_specific_list(input_configuration_specific);

		layer_config_list_modified();
	}

	void network_analyzer::set_input_data(
		const void * input,
		neuron_data_type::input_type type_code,
		unsigned int input_neuron_count)
	{
		// Check schema-reader consistency
		layer_config_list[0].check_equality(static_cast<unsigned int>(input_neuron_count));

		return actual_set_input_data(input, type_code);
	}

	std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> network_analyzer::run_backprop(
		const layer_configuration_specific_snapshot& output_data,
		const std::vector<unsigned int>& output_offset_list,
		unsigned int output_layer_id)
	{
		const layer_configuration_specific& output_config = layer_config_list[output_layer_id + 1];

		std::vector<std::pair<unsigned int, unsigned int> > output_rectangle_borders;
		for(int i = 0; i < output_offset_list.size(); ++i)
			output_rectangle_borders.push_back(std::make_pair(output_offset_list[i], output_offset_list[i] + output_data.config.dimension_sizes[i]));

		std::vector<std::pair<unsigned int, unsigned int> > input_rectangle_borders = schema->get_input_rectangle_borders(output_rectangle_borders, output_layer_id);

		for(unsigned int i = 0; i < input_rectangle_borders.size(); ++i)
			input_rectangle_borders[i].second = std::min(input_rectangle_borders[i].second, layer_config_list[0].dimension_sizes[i]);

		return actual_run_backprop(
			output_data,
			output_offset_list,
			output_layer_id,
			input_rectangle_borders);
	}
}
