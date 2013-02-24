/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "network_updater.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	network_updater::network_updater(
		network_schema_smart_ptr schema,
		const_data_scale_params_smart_ptr scale_params)
		: schema(schema)
		, profile_mode(false)
		, scale_params(scale_params)
	{
	}

	network_updater::~network_updater()
	{
	}

	void network_updater::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
	{
		if ((layer_config_list.size() > 0) && (layer_config_list[0] == input_configuration_specific))
			return;

		if (scale_params == 0)
			current_scale_params = const_data_scale_params_smart_ptr(new data_scale_params(input_configuration_specific.feature_map_count));
		else
		{
			current_scale_params = scale_params;
			if (current_scale_params->feature_map_count != input_configuration_specific.feature_map_count)
				throw neural_network_exception((boost::format("Feature map counts for scaling and in input data don't match: %1% and %2%")
					% current_scale_params->feature_map_count % input_configuration_specific.feature_map_count).str());
		}

		layer_config_list = schema->get_layer_configuration_specific_list(input_configuration_specific);

		update_flops();

		layer_config_list_modified();
	}

	std::vector<testing_result_smart_ptr> network_updater::update(
		supervised_data_reader_byte& reader,
		const std::vector<network_data_smart_ptr>& training_speed_vector_list,
		std::vector<network_data_smart_ptr>& data_list,
		const std::map<unsigned int, float>& layer_to_dropout_rate_map,
		const std::vector<float>& random_uniform_list)
	{
		// Check data-schema consistency
		for(std::vector<network_data_smart_ptr>::iterator it = data_list.begin(); it != data_list.end(); it++)
			(*it)->check_network_data_consistency(*schema);
		for(std::vector<network_data_smart_ptr>::const_iterator it = training_speed_vector_list.begin(); it != training_speed_vector_list.end(); it++)
			(*it)->check_network_data_consistency(*schema);

		set_input_configuration_specific(reader.get_input_configuration());

		// Check schema-reader consistency
		layer_config_list[layer_config_list.size() - 1].check_equality(reader.get_output_configuration());

		return actual_update(reader, training_speed_vector_list, data_list, layer_to_dropout_rate_map, random_uniform_list);
	}

	void network_updater::update_flops()
	{
		flops = 0.0F;
		const const_layer_list& layer_list = *schema;
		bool non_empty_data_encountered = false;
		for(unsigned int i = 0; i < layer_list.size(); i++)
		{
			const_layer_smart_ptr layer = layer_list[i];
			const layer_configuration_specific& layer_conf = layer_config_list[i];

			flops += layer->get_forward_flops(layer_conf);

			if (non_empty_data_encountered)
				flops += layer->get_backward_flops(layer_conf);

			non_empty_data_encountered = non_empty_data_encountered || (!layer->is_empty_data());

			flops += layer->get_weights_update_flops(layer_conf);
		}
	}

	float network_updater::get_flops_for_single_entry() const
	{
		return flops;
	}
}
