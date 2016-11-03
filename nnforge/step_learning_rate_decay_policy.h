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

#include "learning_rate_decay_policy.h"

#include <vector>

namespace nnforge
{
	class step_learning_rate_decay_policy : public learning_rate_decay_policy
	{
	public:
		step_learning_rate_decay_policy(const std::string& semicolon_separated_list);

		virtual ~step_learning_rate_decay_policy() = default;

		virtual float get_learning_rate_decay(unsigned int epoch) const;

	private:
		struct decay_rate_entry
		{
			unsigned int start_epoch;
			float decay_rate;
		};

		std::vector<decay_rate_entry> decay_rate_entry_list;
	};
}
