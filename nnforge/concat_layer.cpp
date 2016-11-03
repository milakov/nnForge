/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "concat_layer.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	const std::string concat_layer::layer_type_name = "Concat";

	std::string concat_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr concat_layer::clone() const
	{
		return layer::ptr(new concat_layer(*this));
	}

	float concat_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		return 0.0F;
	}

	layer_configuration_specific concat_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		unsigned int feature_map_count = input_configuration_specific_list[0].feature_map_count;
		unsigned int neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
		for(std::vector<layer_configuration_specific>::const_iterator it = input_configuration_specific_list.begin() + 1; it != input_configuration_specific_list.end(); ++it)
		{
			feature_map_count += it->feature_map_count;
			unsigned int new_neuron_count_per_feature_map = it->get_neuron_count_per_feature_map();
			if (neuron_count_per_feature_map != new_neuron_count_per_feature_map)
				throw neural_network_exception("Neuron count per feature maps mismatch in 2 input layers for concat_layer");
		}

		return layer_configuration_specific(feature_map_count, input_configuration_specific_list[0].dimension_sizes);
	}

	bool concat_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}
}
