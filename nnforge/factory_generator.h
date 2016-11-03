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

#include "forward_propagation_factory.h"
#include "backward_propagation_factory.h"
#include "config_options.h"

#include <memory>

namespace nnforge
{
	class factory_generator
	{
	public:
		typedef std::shared_ptr<factory_generator> ptr;

		virtual ~factory_generator() = default;

		virtual void initialize() = 0;

		virtual forward_propagation_factory::ptr create_forward_propagation_factory() const = 0;

		virtual backward_propagation_factory::ptr create_backward_propagation_factory() const = 0;

		virtual void info() const = 0;

		virtual std::vector<string_option> get_string_options();

		virtual std::vector<multi_string_option> get_multi_string_options();

		virtual std::vector<path_option> get_path_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

	protected:
		factory_generator() = default;
	};
}
