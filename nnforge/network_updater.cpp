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

#include "network_updater.h"

#include "neural_network_exception.h"
#include "rnd.h"
#include "nn_types.h"

#include <boost/format.hpp>

namespace nnforge
{
	network_updater::network_updater(
		network_schema_smart_ptr schema,
		const_error_function_smart_ptr ef)
		: schema(schema)
		, ef(ef)
	{
		const const_layer_list& layers = *schema;
		for(unsigned int i = 0; i < layers.size(); ++i)
		{
			tiling_factor new_tf = layers[i]->get_tiling_factor();
			if (new_tf != 1)
				throw neural_network_exception((boost::format("network_updater cannot run layer %1% with non-unit tiling factor") % i).str());
		}
	}

	network_updater::~network_updater()
	{
	}

	void network_updater::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
	{
		if ((layer_config_list.size() > 0) && (layer_config_list[0] == input_configuration_specific))
			return;

		layer_config_list = schema->get_layer_configuration_specific_list(input_configuration_specific);

		update_flops();

		layer_config_list_modified();
	}

	std::pair<testing_result_smart_ptr, training_stat_smart_ptr> network_updater::update(
		supervised_data_reader& reader,
		const std::vector<std::vector<float> >& learning_rates,
		network_data_smart_ptr data,
		network_data_smart_ptr momentum_data,
		unsigned int batch_size,
		float weight_decay,
		training_momentum momentum,
		bool deterministic_only)
	{
		// Check data-schema consistency
		data->check_network_data_consistency(*schema);

		set_input_configuration_specific(reader.get_input_configuration());

		// Check schema-reader consistency
		layer_config_list[layer_config_list.size() - 1].check_equality(reader.get_output_configuration());

		std::pair<testing_result_smart_ptr, training_stat_smart_ptr> res = actual_update(
			reader,
			learning_rates,
			data,
			momentum_data,
			batch_size,
			weight_decay,
			momentum,
			deterministic_only);

		return res;
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
