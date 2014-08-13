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

#include "testing_result.h"
#include "network_data.h"

#include <boost/chrono.hpp>

namespace nnforge
{
	class training_task_state
	{
	public:
		training_task_state();

		unsigned int index_peeked;
		network_data_smart_ptr data;
		std::vector<testing_result_smart_ptr> history;
		std::vector<std::string> comments;
		unsigned int initial_epoch;

		unsigned int get_current_epoch() const
		{
			return static_cast<unsigned int>(history.size() + initial_epoch);
		}
	};
}
