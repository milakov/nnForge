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

#include "mse_error_function.h"

namespace nnforge
{
	// {2ABB68BB-6EE8-48F4-B530-5F9E3AECC2E2}
	const boost::uuids::uuid mse_error_function::function_guid =
		{ 0x2a, 0xbb, 0x68, 0xbb
		, 0x6e, 0xe8
		, 0x48, 0xf4
		, 0xb5, 0x30
		, 0x5f, 0x9e, 0x3a, 0xec, 0xc2, 0xe2 };

	mse_error_function::mse_error_function()
	{
	}

	mse_error_function::~mse_error_function()
	{
	}

	const boost::uuids::uuid& mse_error_function::get_uuid() const
	{
		return function_guid;
	}

	std::string mse_error_function::get_function_name() const
	{
		return "MSE";
	}

	float mse_error_function::calculate_error(
		const float * actual_values,
		const float * predicted_values,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float diff = actual_values[i] - predicted_values[i];
			sum += diff * diff;
		}

		return sum * 0.5F;
	}

	float mse_error_function::calculate_gradient_and_error(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float diff = actual_values[i] - predicted_values[i];
			sum += diff * diff;
			gradient[i] = diff;
		}

		return sum * 0.5F;
	}
}
