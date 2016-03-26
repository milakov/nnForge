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

#include "profile_state.h"
#include "layer_name_with_action.h"

#include <map>

namespace nnforge
{
	class profile_util
	{
	public:
		static void dump_layer_action_performance(
			profile_state::ptr profile,
			float max_flops,
			const char * action_prefix,
			unsigned int entry_count,
			const std::map<layer_name_with_action, float>& action_flops_per_entry,
			const std::map<layer_name_with_action, float>& action_seconds,
			const std::map<std::string, std::string>& layer_name_to_layer_type_map,
			float total_seconds);

	private:
		struct entry
		{
			entry(layer_name_with_action action, float seconds)
				: action(action)
				, seconds(seconds)
			{
			};

			layer_name_with_action action;
			float seconds;
		};

		static bool compare_entry(const entry& i, const entry& j);

	private:
		profile_util();
		~profile_util();
	};
}
