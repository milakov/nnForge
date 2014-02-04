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

#include "network_trainer.h"

#include <vector>

#include "neural_network_exception.h"

namespace nnforge
{
	network_trainer::network_trainer(network_schema_smart_ptr schema)
		: schema(schema)
		, epoch_count(50)
		, learning_rate_decay_tail_epoch_count(0)
		, learning_rate_decay_rate(0.5F)
	{
	}

	network_trainer::~network_trainer()
	{
	}

	void network_trainer::train(
		supervised_data_reader& reader,
		network_data_peeker& peeker,
		network_data_pusher& progress_pusher,
		network_data_pusher& pusher)
	{
		initialize_train(reader);
		unsigned int max_batch_size = get_max_batch_size();

		if (max_batch_size == 0)
			throw neural_network_exception("The trainer is unable to train even a single network");

		std::vector<training_task_state> task_list;

		while(true)
		{
			while (task_list.size() < max_batch_size)
			{
				training_task_state new_task;
				std::pair<unsigned int, network_data_smart_ptr> data_with_key = peeker.peek(schema);
				if (data_with_key.second == 0)
					break;

				new_task.index_peeked = data_with_key.first;
				new_task.data = data_with_key.second;
				task_list.push_back(new_task);
			}

			if (task_list.size() == 0)
				break; // Nothing is left to be trained

			train_step(
				reader,
				task_list);

			for(int i = 0; i < task_list.size(); ++i)
				progress_pusher.push(task_list[i]);

			for(int i = static_cast<int>(task_list.size()) - 1; i >= 0; --i)
			{
				if (is_broken(task_list[i]))
				{
					std::cout << "# " << task_list[i].index_peeked << " - broken weights while training, discarding it." << std::endl;
					task_list.erase(task_list.begin() + i);
					continue;
				}

				if (is_last_epoch(task_list[i]))
				{
					pusher.push(task_list[i]);
					task_list.erase(task_list.begin() + i);
				}
			}
		}
	}

	bool network_trainer::is_last_epoch(const training_task_state& state) const
	{
		return (state.history.size() >= epoch_count);
	}

	bool network_trainer::is_broken(const training_task_state& state) const
	{
		float mse = state.history.back()->get_mse();
		bool sanity_check = (mse < 1.0e+10F) && (-mse > -1.0E+10F) && !(-mse < -1.0E+10F);
		return !sanity_check;
	}

	float network_trainer::get_tail_decay_factor(unsigned int epoch) const
	{
		int first_iteration_with_decay = std::max(static_cast<int>(epoch_count) - static_cast<int>(learning_rate_decay_tail_epoch_count), 1);
		int tail_degradation_epoch = static_cast<int>(epoch) - first_iteration_with_decay + 1;
		if (tail_degradation_epoch <= 0)
			return 1.0F;
		else
			return powf(learning_rate_decay_rate, static_cast<float>(tail_degradation_epoch));
	}
}
