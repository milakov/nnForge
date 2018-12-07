/*
 *  Copyright 2011-2018 Maxim Milakov
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

#include "step_learning_rate_decay_policy.h"

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <algorithm>

#include "neural_network_exception.h"

namespace nnforge
{
	step_learning_rate_decay_policy::step_learning_rate_decay_policy(
		const std::string& semicolon_separated_list,
		int step_learning_rate_warmup_epochs)
	{
		this->step_learning_rate_warmup_epochs = step_learning_rate_warmup_epochs;

		std::vector<std::string> strs;
		if (!semicolon_separated_list.empty())
			boost::split(strs, semicolon_separated_list, boost::is_any_of(":"));
		if (strs.size() % 2 != 0)
			throw neural_network_exception((boost::format("Invalid initialization string for step learning rate decay policy: %1%") % semicolon_separated_list).str());

		for(int i = 0; i < strs.size(); i += 2)
		{
			decay_rate_entry new_entry;
			new_entry.start_epoch = atol(strs[i].c_str());
			new_entry.decay_rate = static_cast<float>(atof(strs[i + 1].c_str()));
			decay_rate_entry_list.push_back(new_entry);
		}
		std::sort(decay_rate_entry_list.begin(), decay_rate_entry_list.end(), [] (const decay_rate_entry& i, const decay_rate_entry& j) { return (i.start_epoch < j.start_epoch); });
	}

	float step_learning_rate_decay_policy::get_learning_rate_decay(unsigned int epoch) const
	{
		float current_decay = 1.0F;
		for(int i = 0; i < decay_rate_entry_list.size(); ++i)
		{
			if (epoch >= decay_rate_entry_list[i].start_epoch)
				current_decay = decay_rate_entry_list[i].decay_rate;
			else
				break;
		}
		if (epoch < step_learning_rate_warmup_epochs - 1)
		{
			current_decay *= static_cast<float>(epoch + 1) / static_cast<float>(step_learning_rate_warmup_epochs);
		}
		return current_decay;
	}
}
