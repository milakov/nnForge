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

#include "report_progress_network_data_pusher.h"

#include <stdio.h>
#include <boost/format.hpp>

namespace nnforge
{
	report_progress_network_data_pusher::report_progress_network_data_pusher()
	{
	}

	report_progress_network_data_pusher::~report_progress_network_data_pusher()
	{
	}

	void report_progress_network_data_pusher::push(const training_task_state& task_state)
	{
		unsigned int last_index = static_cast<unsigned int>(task_state.history.size()) - 1;

		std::cout << "# " << task_state.index_peeked
			<< ", It " << (last_index + 1)
			<< ", Training "
			<< *(task_state.history[last_index]);

		if (last_index > 0)
		{
			std::cout << " (";

			float mse_improvement = task_state.history[last_index - 1]->get_mse() - task_state.history[last_index]->get_mse();
			std::cout << (boost::format("Imp %|1$.6f|") % mse_improvement);

			if (last_index > 1)
			{
				float previous_mse_improvement = task_state.history[last_index - 2]->get_mse() - task_state.history[last_index - 1]->get_mse();
				if ((previous_mse_improvement > 0.0F) && (mse_improvement >= 0.0F))
				{
					float improvement_ratio = mse_improvement / previous_mse_improvement * 100.0F;
					std::cout << (boost::format(", Ratio %|1$.0f|%%") % improvement_ratio);
				}
			}

			std::cout << ")";
		}

		if (task_state.comments[last_index].size() > 0)
			std::cout << ", " << task_state.comments[last_index];

		std::cout << std::endl;
	}
}
