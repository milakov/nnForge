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

#include "error_function.h"

#include "neural_network_exception.h"

namespace nnforge
{
	boost::uuids::uuid error_function::empty_guid = boost::uuids::uuid();

	error_function::error_function()
	{
	}

	error_function::~error_function()
	{
	}

	const boost::uuids::uuid& error_function::get_fusable_activation_uuid() const
	{
		return empty_guid;
	}

	float error_function::calculate_gradient_and_error_fused_with_activation(
		const float * actual_values,
		const float * predicted_values,
		float * gradient,
		unsigned int neuron_count) const
	{
		throw neural_network_exception("calculate_gradient_and_error_fused_with_activation not implemented for this error function");
	}
}
