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

#include "cross_entropy_error_function.h"

#include <cmath>

namespace nnforge
{
	// {7E4D6B3E-39D6-4E85-8049-5CE57D3E0FD2}
	const boost::uuids::uuid cross_entropy_error_function::function_guid =
		{ 0x7e, 0x4d, 0x6b, 0x3e
		, 0x39, 0xd6
		, 0x4e, 0x85
		, 0x80, 0x49
		, 0x5c, 0xe5, 0x7d, 0x3e, 0xf, 0xd2 };

	cross_entropy_error_function::cross_entropy_error_function()
	{
	}

	cross_entropy_error_function::~cross_entropy_error_function()
	{
	}

	const boost::uuids::uuid& cross_entropy_error_function::get_uuid() const
	{
		return function_guid;
	}

	std::string cross_entropy_error_function::get_function_name() const
	{
		return "CE";
	}

	float cross_entropy_error_function::calculate_error(
		const float * actual_values,
		const float * predicted_values,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float actual_val = actual_values[i];
			if (actual_val > 0.0F)
				sum -= actual_val * logf(predicted_values[i]);
			if (actual_val < 1.0F)
				sum -= (1.0F - actual_val) * logf(1.0F - predicted_values[i]);
		}

		return sum;
	}

	void cross_entropy_error_function::calculate_gradient(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float actual_val = actual_values[i];
			float gradient_val = 0.0F;
			if (actual_val > 0.0F)
				gradient_val += actual_val / predicted_values[i];
			if (actual_val < 1.0F)
				gradient_val -= (1.0F - actual_val) / (1.0F - predicted_values[i]);

			gradient[i] = gradient_val;
		}
	}
}
