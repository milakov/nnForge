/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "network_tester_factory.h"
#include "network_updater_factory.h"
#include "hessian_calculator_factory.h"
#include "config_options.h"

namespace nnforge
{
	class factory_generator
	{
	public:
		virtual ~factory_generator();

		virtual void initialize() = 0;

		virtual network_tester_factory_smart_ptr create_tester_factory() const = 0;

		virtual network_updater_factory_smart_ptr create_updater_factory() const = 0;

		virtual hessian_calculator_factory_smart_ptr create_hessian_factory() const = 0;

		virtual void info() const = 0;

		virtual std::vector<string_option> get_string_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

	protected:
		factory_generator();
	};

	typedef std::tr1::shared_ptr<factory_generator> factory_generator_smart_ptr;
}
