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
		, learning_rate(0.02F)
		, batch_size(1)
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
		unsigned int reader_epoch_id = 0;

		initialize_train(reader);

		while(true)
		{
			network_data_peek_entry entry_peeked = peeker.peek(schema);
			if (entry_peeked.data == 0)
				break;

			training_task_state new_task;
			new_task.index_peeked = entry_peeked.index;
			new_task.data = entry_peeked.data;
			new_task.initial_epoch = entry_peeked.start_epoch;

			bool empty_momentum = false;
			if (momentum.type == training_momentum::no_momentum)
				new_task.momentum_data = network_data_smart_ptr();
			else
			{
				if (entry_peeked.momentum_data)
					new_task.momentum_data = entry_peeked.momentum_data;
				else
				{
					new_task.momentum_data = network_data_smart_ptr(new network_data(*schema));
					if (new_task.initial_epoch > 0)
						empty_momentum = true;
				}
			}
			
			if (is_last_epoch(new_task))
			{
				std::cout << "Warning: Task is allocated which is already complete. Index " << new_task.index_peeked << ", Base epoch " << new_task.initial_epoch << std::endl;
				continue;
			}

			std::cout << "New task allocated: Index " << new_task.index_peeked << ", Base epoch " << new_task.initial_epoch;
			if (empty_momentum)
				std::cout << ", Starting with empty momentum";
			std::cout << std::endl;

			if (new_task.initial_epoch > reader_epoch_id)
			{
				for(unsigned int i = reader_epoch_id; i < new_task.initial_epoch; ++i)
					reader.next_epoch();
				reader_epoch_id += (new_task.initial_epoch - reader_epoch_id);
			}
			else if (new_task.initial_epoch < reader_epoch_id)
				std::cout << "Warning: negative scrolling through reader requested. Index " << new_task.index_peeked << ", Initial epoch " << new_task.initial_epoch << std::endl;

			while(true)
			{
				train_step(
					reader,
					new_task);

				reader.next_epoch();
				++reader_epoch_id;

				progress_pusher.push(new_task);

				if (is_broken(new_task))
				{
					std::cout << "# " << new_task.index_peeked << " - broken weights while training, discarding it." << std::endl;
					break;
				}

				if (is_last_epoch(new_task))
				{
					pusher.push(new_task);
					break;
				}
			}
		}
	}

	bool network_trainer::is_last_epoch(const training_task_state& state) const
	{
		return (state.get_current_epoch() >= epoch_count);
	}

	bool network_trainer::is_broken(const training_task_state& state) const
	{
		float error = state.history.back().first->get_error();
		bool sanity_check = (error < 1.0e+10F) && (-error > -1.0E+10F) && !(-error < -1.0E+10F);
		return !sanity_check;
	}

	float network_trainer::get_global_learning_rate(unsigned int epoch) const
	{
		float tail_degradation_factor = 1.0F;
		{
			int first_iteration_with_decay = std::max(static_cast<int>(epoch_count) - static_cast<int>(learning_rate_decay_tail_epoch_count), 1);
			int tail_degradation_epoch = static_cast<int>(epoch) - first_iteration_with_decay + 1;
			if (tail_degradation_epoch > 0)
				tail_degradation_factor = powf(learning_rate_decay_rate, static_cast<float>(tail_degradation_epoch));
		}

		float head_degradation_factor = 1.0F;
		{
			int head_rise_epoch = static_cast<int>(learning_rate_rise_head_epoch_count) - static_cast<int>(epoch);
			if (head_rise_epoch > 0)
				head_degradation_factor = powf(learning_rate_rise_rate, static_cast<float>(head_rise_epoch));
		}

		return tail_degradation_factor * head_degradation_factor * learning_rate;
	}
}
