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

#include "network_tester.h"

#include "neural_network_exception.h"
#include <boost/chrono.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	network_tester::network_tester(
		network_schema_smart_ptr schema,
		const_data_scale_params_smart_ptr scale_params)
		: schema(schema)
		, scale_params(scale_params)
	{
	}

	network_tester::~network_tester()
	{
	}

	void network_tester::set_data(network_data_smart_ptr data)
	{
		// Check data-schema consistency
		data->check_network_data_consistency(*schema);

		actual_set_data(data);
	}

	void network_tester::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
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

	void network_tester::test(
		supervised_data_reader_byte& reader,
		testing_complete_result_set& result)
	{
		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();

		set_input_configuration_specific(reader.get_input_configuration());

		// Check schema-reader consistency
		layer_config_list[layer_config_list.size() - 1].check_equality(reader.get_output_configuration());

		actual_test(reader, result);

		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;

		result.mse->flops = static_cast<float>(result.mse->entry_count) * flops;
		result.mse->time_to_complete_seconds = sec.count();
	}

	output_neuron_value_set_smart_ptr network_tester::run(unsupervised_data_reader_byte& reader)
	{
		set_input_configuration_specific(reader.get_input_configuration());

		return actual_run(reader);
	}

	std::vector<layer_configuration_specific_snapshot_smart_ptr> network_tester::get_snapshot(std::vector<unsigned char>& input)
	{
		// Check schema-reader consistency
		layer_config_list[0].check_equality(static_cast<unsigned int>(input.size()));

		return actual_get_snapshot(input);
	}

	layer_configuration_specific_snapshot_smart_ptr network_tester::run(std::vector<unsigned char>& input)
	{
		// Check schema-reader consistency
		layer_config_list[0].check_equality(static_cast<unsigned int>(input.size()));

		return actual_run(input);
	}

	void network_tester::update_flops()
	{
		flops = 0.0F;
		const const_layer_list& layer_list = *schema;
		for(unsigned int i = 0; i < layer_list.size(); i++)
		{
			flops += layer_list[i]->get_forward_flops(layer_config_list[i]);
		}
	}

	float network_tester::get_flops_for_single_entry() const
	{
		return flops;
	}
}
