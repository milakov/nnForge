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

#pragma once

#include "error_function.h"

// E = sum((x_i - y_i)^2) / 2
namespace nnforge
{
	class mse_error_function : public error_function
	{
	public:
		mse_error_function();

		virtual ~mse_error_function();

		virtual const boost::uuids::uuid& get_uuid() const;

		virtual std::string get_function_name() const;

		virtual float calculate_error(
			const float * actual_values,
			const float * predicted_values,
			unsigned int neuron_count) const;

		virtual float calculate_gradient_and_error(
			const float * actual_values,
			const float * predicted_values,
			float * gradient,
			unsigned int neuron_count) const;

		static const boost::uuids::uuid function_guid;
	};
}
