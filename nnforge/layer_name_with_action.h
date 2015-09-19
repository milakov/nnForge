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

#include "layer_action.h"

#include <string>

namespace nnforge
{
	class layer_name_with_action
	{
	public:
		layer_name_with_action()
		{
		}

		layer_name_with_action(
			const std::string& name,
			layer_action action)
			: name(name)
			, action(action)
		{
		}

		const std::string& get_name() const
		{
			return name;
		}

		const layer_action& get_action() const
		{
			return action;
		}

	private:
		std::string name;
		layer_action action;

		friend bool operator <(const layer_name_with_action& x, const layer_name_with_action& y);
	};

	inline bool operator <(const layer_name_with_action& x, const layer_name_with_action& y)
	{
		if (x.name < y.name)
		{
			return true;
		}
		else if (y.name < x.name)
		{
			return false;
		}
		else
		{
			return x.action < y.action;
		}
	}
}
