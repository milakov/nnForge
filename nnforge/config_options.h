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

#include <string>

namespace nnforge
{
	struct string_option
	{
		string_option(
			const char * name,
			std::string * var,
			const char * default_value,
			const char * description)
			: name(name)
			, var(var)
			, default_value(default_value)
			, description(description)
		{
		}

		std::string name;
		std::string * var;
		std::string default_value;
		std::string description;
	};

	struct bool_option
	{
		bool_option(
			const char * name,
			bool * var,
			bool default_value,
			const char * description)
			: name(name)
			, var(var)
			, default_value(default_value)
			, description(description)
		{
		}

		std::string name;
		bool * var;
		bool default_value;
		std::string description;
	};

	struct float_option
	{
		float_option(
			const char * name,
			float * var,
			float default_value,
			const char * description)
			: name(name)
			, var(var)
			, default_value(default_value)
			, description(description)
		{
		}

		std::string name;
		float * var;
		float default_value;
		std::string description;
	};

	struct int_option
	{
		int_option(
			const char * name,
			int * var,
			int default_value,
			const char * description)
			: name(name)
			, var(var)
			, default_value(default_value)
			, description(description)
		{
		}

		std::string name;
		int * var;
		int default_value;
		std::string description;
	};
}

