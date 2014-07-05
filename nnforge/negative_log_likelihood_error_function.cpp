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

#include "negative_log_likelihood_error_function.h"

#include "softmax_layer.h"

#include <cmath>

namespace nnforge
{
	// {1955940E-D494-4776-BFAA-48974D9652F2}
	const boost::uuids::uuid negative_log_likelihood_error_function::function_guid =
		{ 0x19, 0x55, 0x94, 0x0e
		, 0xd4, 0x94
		, 0x47, 0x76
		, 0xbf, 0xaa
		, 0x48, 0x97, 0x4d, 0x96, 0x52, 0xf2 };

	negative_log_likelihood_error_function::negative_log_likelihood_error_function()
	{
	}

	negative_log_likelihood_error_function::~negative_log_likelihood_error_function()
	{
	}

	const boost::uuids::uuid& negative_log_likelihood_error_function::get_uuid() const
	{
		return function_guid;
	}

	std::string negative_log_likelihood_error_function::get_function_name() const
	{
		return "NLL";
	}

	float negative_log_likelihood_error_function::calculate_error(
		const float * actual_values,
		const float * predicted_values,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float actual_val = actual_values[i];
			if (actual_val > 0.0F)
				sum -= actual_val * logf(std::max(predicted_values[i], 1.0e-20F));
		}

		return sum;
	}

	float negative_log_likelihood_error_function::calculate_gradient_and_error(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float actual_val = actual_values[i];
			float gradient_val = 0.0F;
			if (actual_val > 0.0F)
			{
				sum -= actual_val * logf(std::max(predicted_values[i], 1.0e-20F));
				gradient_val = actual_val / predicted_values[i];
			}

			gradient[i] = gradient_val;
		}

		return sum;
	}

	// Works correctly only when:
	// 1) Actual_values are all 0 except for single value which is 1
	// 2) Feature map contains exactly 1 element
	float negative_log_likelihood_error_function::calculate_gradient_and_error_fused_with_activation(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		float predicted_sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
			predicted_sum += expf(predicted_values[i]);

		float mult = 1.0F / predicted_sum;

		float error_sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float actual_val = actual_values[i];
			float predicted_val = expf(predicted_values[i]) * mult;
			gradient[i] = actual_val - predicted_val;
			if (actual_val > 0.0F)
				error_sum -= actual_val * logf(std::max(predicted_val, 1.0e-20F));
		}

		return error_sum;
	}

	const boost::uuids::uuid& negative_log_likelihood_error_function::get_fusable_activation_uuid() const
	{
		return softmax_layer::layer_guid;
	}
}
