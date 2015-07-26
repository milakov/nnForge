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

#include "softmax_layer.h"
#include "layer_factory.h"

namespace nnforge
{
	const std::string softmax_layer::layer_type_name = "Softmax";

	softmax_layer::softmax_layer()
	{
	}

	std::string softmax_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr softmax_layer::clone() const
	{
		return layer::ptr(new softmax_layer(*this));
	}

	float softmax_layer::get_forward_flops(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		return static_cast<float>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * (input_configuration_specific_list[0].feature_map_count * 3 - 1));
	}

	float softmax_layer::get_backward_flops(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		unsigned int input_layer_id) const
	{
		return static_cast<float>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * (input_configuration_specific_list[0].feature_map_count * 4 - 1));
	}
}
