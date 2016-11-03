/*
 *  Copyright 2011-2016 Maxim Milakov
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
#include <boost/format.hpp>

namespace nnforge
{
	class layer_action
	{
	public:
		enum action_type
		{
			forward = 0,
			backward_data = 1,
			backward_weights = 2,
			backward_data_and_weights = 3,
			update_weights = 4
		};

		layer_action() = default;

		layer_action(
			action_type at,
			int backprop_index = -1)
			: at(at)
			, backprop_index(backprop_index)
		{
		}

		action_type get_action_type() const
		{
			return at;
		}

		int get_backprop_index() const
		{
			return backprop_index;
		}

		std::string str() const
		{
			if (at == forward)
			{
				return "forward";
			}
			else if (at == backward_weights)
			{
				return "backward_weights";
			}
			else if (at == backward_data_and_weights)
			{
				return "backward_data_and_weights";
			}
			else if (at == update_weights)
			{
				return "update_weights";
			}
			else
			{
				return (boost::format("backward_data_%1%") % backprop_index).str();
			}
		}

	private:
		action_type at;
		int backprop_index;

		friend bool operator <(const layer_action& x, const layer_action& y);
	};

	inline bool operator <(const layer_action& x, const layer_action& y)
	{
		return (((unsigned long long)x.at << 32) | (unsigned int)x.backprop_index) < (((unsigned long long)y.at << 32) | (unsigned int)y.backprop_index);
	}
}
