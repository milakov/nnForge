/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "hessian_calculator.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	hessian_calculator::hessian_calculator(network_schema_smart_ptr schema)
		: schema(schema)
	{
	}

	hessian_calculator::~hessian_calculator()
	{
	}

	void hessian_calculator::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
	{
		if ((layer_config_list.size() > 0) && (layer_config_list[0] == input_configuration_specific))
			return;

		layer_config_list = schema->get_layer_configuration_specific_list(input_configuration_specific);

		update_flops();

		layer_config_list_modified();
	}

	layer_data_list_smart_ptr hessian_calculator::get_hessian(
		unsupervised_data_reader& reader,
		network_data_smart_ptr data,
		unsigned int hessian_entry_to_process_count)
	{
		set_input_configuration_specific(reader.get_input_configuration());

		// Check data-schema consistency
		data->check_network_data_consistency(*schema);

		return actual_get_hessian(
			reader,
			data,
			hessian_entry_to_process_count);
	}

	void hessian_calculator::update_flops()
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
				flops += layer->get_backward_flops_2nd(layer_conf);

			non_empty_data_encountered = non_empty_data_encountered || (!layer->is_empty_data());

			flops += layer->get_weights_update_flops_2nd(layer_conf);
		}
	}

	float hessian_calculator::get_flops_for_single_entry() const
	{
		return flops;
	}
}
