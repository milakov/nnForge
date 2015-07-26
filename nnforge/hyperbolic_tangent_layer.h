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

namespace nnforge
{
	// f(x) = scale * tanh(steepness * x)
	// tanh(x) = (exp(2 * x) - 1) / (exp(2 * x) + 1)
	// Derivative:
	// f'(x) = scale * steepness * (1 - (f(x) / scale)^2)
	// Typical values:
	// 1) scale = 1, steepness = 1
	// 2) scale = 1.7159, steepness = 0.666666F
	class hyperbolic_tangent_layer : public layer
	{
	public:
		hyperbolic_tangent_layer(
			float scale = 1.0F,
			float steepness = 1.0F);

		virtual layer::ptr clone() const;

		virtual float get_forward_flops(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const;

		virtual float get_backward_flops(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			unsigned int input_layer_id) const;

		virtual std::string get_type_name() const;

		virtual void write_proto(void * layer_proto) const;

		virtual void read_proto(const void * layer_proto);

		static const std::string layer_type_name;

	public:
		float scale;
		float steepness;
	};
}
