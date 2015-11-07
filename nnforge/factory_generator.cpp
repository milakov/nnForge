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

#include "factory_generator.h"

namespace nnforge
{
	factory_generator::factory_generator()
	{
	}

	factory_generator::~factory_generator()
	{
	}

	std::vector<string_option> factory_generator::get_string_options()
	{
		return std::vector<string_option>();
	}

	std::vector<multi_string_option> factory_generator::get_multi_string_options()
	{
		return std::vector<multi_string_option>();
	}

	std::vector<path_option> factory_generator::get_path_options()
	{
		return std::vector<path_option>();
	}

	std::vector<bool_option> factory_generator::get_bool_options()
	{
		return std::vector<bool_option>();
	}

	std::vector<float_option> factory_generator::get_float_options()
	{
		return std::vector<float_option>();
	}

	std::vector<int_option> factory_generator::get_int_options()
	{
		return std::vector<int_option>();
	}
}
