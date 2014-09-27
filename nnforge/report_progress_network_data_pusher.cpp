/*
 *  Copyright 2011-2014 Maxim Milakov
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
			<< ", Epoch " << task_state.get_current_epoch()
			<< ", Training "
			<< *(task_state.history[last_index].first);

		if (last_index > 0)
		{
			std::cout << " (";

			float error_improvement = task_state.history[last_index - 1].first->get_error() - task_state.history[last_index].first->get_error();
			std::cout << (boost::format("Imp %|1$.6f|") % error_improvement);

			if (last_index > 1)
			{
				float previous_error_improvement = task_state.history[last_index - 2].first->get_error() - task_state.history[last_index - 1].first->get_error();
				if ((previous_error_improvement > 0.0F) && (error_improvement >= 0.0F))
				{
					float improvement_ratio = error_improvement / previous_error_improvement * 100.0F;
					std::cout << (boost::format(", Ratio %|1$.0f|%%") % improvement_ratio);
				}
			}

			std::cout << ")";
		}

		if (task_state.comments[last_index].size() > 0)
			std::cout << ", " << task_state.comments[last_index];

		std::cout << ", Avg [rate weights updates]";
		for(int layer_id = 0; layer_id < task_state.data->data_list.size(); ++layer_id)
		{
			layer_data_smart_ptr layer_data = task_state.data->data_list[layer_id];
			if (!layer_data->empty())
			{
				std::cout << " #" << layer_id;
				const std::vector<float>& absolute_updates = task_state.history[last_index].second->absolute_updates[layer_id];
				for(int part_id = 0; part_id < layer_data->size(); ++part_id)
				{
					const std::vector<float>& weights = layer_data->at(part_id);
					double sum = 0.0;
					for(std::vector<float>::const_iterator it = weights.begin(); it != weights.end(); ++it)
						sum += static_cast<double>(fabsf(*it));
					float avg_weight = static_cast<float>(sum) / static_cast<float>(weights.size());

					std::cout << (boost::format(" [%|1$.2e| %|2$.2e| %|3$.2e|]") % (absolute_updates[part_id] / avg_weight) % avg_weight % absolute_updates[part_id]); 
				}
			}
		}

		std::cout << std::endl;
	}
}
