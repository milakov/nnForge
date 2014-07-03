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

#include "nn_types.h"

#include <string>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class error_function
	{
	public:
		virtual ~error_function();

		virtual const boost::uuids::uuid& get_uuid() const = 0;

		virtual std::string get_function_name() const = 0;

		virtual float calculate_error(
			const float * actual_values,
			const float * predicted_values,
			unsigned int neuron_count) const = 0;

		virtual float calculate_gradient_and_error(
			const float * actual_values,
			const float * predicted_values,
			float * gradient,
			unsigned int neuron_count) const = 0;

		virtual float calculate_gradient_and_error_fused_with_activation(
			const float * actual_values,
			const float * predicted_values,
			float * gradient,
			unsigned int neuron_count) const;

		virtual const boost::uuids::uuid& get_fusable_activation_uuid() const;

	protected:
		error_function();

	private:
		static boost::uuids::uuid empty_guid;
	};

	typedef nnforge_shared_ptr<error_function> error_function_smart_ptr;
	typedef nnforge_shared_ptr<const error_function> const_error_function_smart_ptr;
}
