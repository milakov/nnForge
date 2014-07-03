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

#include "squared_hinge_loss_error_function.h"

namespace nnforge
{
	// {406DBE54-CF84-4135-871D-936E94F2B175}
	const boost::uuids::uuid squared_hinge_loss_error_function::function_guid =
		{ 0x40, 0x6d, 0xbe, 0x54
		, 0xcf, 0x84
		, 0x41, 0x35
		, 0x87, 0x1d
		, 0x93, 0x6e, 0x94, 0xf2, 0xb1, 0x75 };

	squared_hinge_loss_error_function::squared_hinge_loss_error_function()
	{
	}

	squared_hinge_loss_error_function::~squared_hinge_loss_error_function()
	{
	}

	const boost::uuids::uuid& squared_hinge_loss_error_function::get_uuid() const
	{
		return function_guid;
	}

	std::string squared_hinge_loss_error_function::get_function_name() const
	{
		return "SHL";
	}

	float squared_hinge_loss_error_function::calculate_error(
		const float * actual_values,
		const float * predicted_values,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float diff = ((actual_values[i] > 0.0F) && (predicted_values[i] < actual_values[i])) || ((actual_values[i] <= 0.0F) && (predicted_values[i] > actual_values[i])) ? actual_values[i] - predicted_values[i] : 0.0F;
			sum += diff * diff;
		}

		return sum * 0.5F;
	}

	float squared_hinge_loss_error_function::calculate_gradient_and_error(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		float sum = 0.0F;
		for(unsigned int i = 0; i < neuron_count; ++i)
		{
			float diff = ((actual_values[i] > 0.0F) && (predicted_values[i] < actual_values[i])) || ((actual_values[i] <= 0.0F) && (predicted_values[i] > actual_values[i])) ? actual_values[i] - predicted_values[i] : 0.0F;
			sum += diff * diff;
			gradient[i] = diff;
		}

		return sum * 0.5F;
	}
}
