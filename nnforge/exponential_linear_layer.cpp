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

#include "exponential_linear_layer.h"

namespace nnforge
{
	const std::string exponential_linear_layer::layer_type_name = "ELU";

	std::string exponential_linear_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr exponential_linear_layer::clone() const
	{
		return layer::ptr(new exponential_linear_layer(*this));
	}

	float exponential_linear_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count() * 2);
		case layer_action::backward_data:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count());
		default:
			return 0.0F;
		}
	}
}
