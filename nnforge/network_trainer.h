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

#pragma once

#include "network_data_peeker.h"
#include "network_data_pusher.h"
#include "training_task_state.h"
#include "network_schema.h"
#include "supervised_data_reader.h"
#include "nn_types.h"

#include <map>

namespace nnforge
{
	class network_trainer
	{
	public:
		virtual ~network_trainer();

		void train(
			supervised_data_reader& reader,
			network_data_peeker& peeker,
			network_data_pusher& progress_pusher,
			network_data_pusher& pusher);

		unsigned int epoch_count;
		unsigned int batch_size;
		float learning_rate;
		unsigned int learning_rate_decay_tail_epoch_count;
		float learning_rate_decay_rate;
		unsigned int learning_rate_rise_head_epoch_count;
		float learning_rate_rise_rate;
		float weight_decay;
		float momentum;
		std::map<unsigned int, float> layer_to_dropout_rate_map;

	protected:
		network_trainer(network_schema_smart_ptr schema);

		float get_global_learning_rate(unsigned int epoch) const;

		virtual void initialize_train(supervised_data_reader& reader) = 0;

		// The method should add testing result to the training history of each element
		virtual void train_step(
			supervised_data_reader& reader,
			training_task_state& task) = 0;

		network_schema_smart_ptr schema;

	private:
		bool is_last_epoch(const training_task_state& state) const;

		bool is_broken(const training_task_state& state) const;

	private:
		network_trainer(const network_trainer&);
		network_trainer& operator =(const network_trainer&);
	};

	typedef nnforge_shared_ptr<network_trainer> network_trainer_smart_ptr;
}
