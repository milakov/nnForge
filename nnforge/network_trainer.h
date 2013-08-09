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

#include "network_data_peeker.h"
#include "network_data_pusher.h"
#include "training_task_state.h"

#include "network_schema.h"
#include "supervised_data_reader.h"

#include <memory>
#include <map>

namespace nnforge
{
	class network_trainer
	{
	public:
		virtual ~network_trainer();

		// If the layer is not in layer_to_dropout_rate_map then its droput rate is assumed to be 0.0F
		void train(
			supervised_data_reader& reader,
			network_data_peeker& peeker,
			network_data_pusher& progress_pusher,
			network_data_pusher& pusher);

		unsigned int iteration_count;

	protected:
		network_trainer(network_schema_smart_ptr schema);

		virtual void initialize_train(supervised_data_reader& reader) = 0;

		virtual unsigned int get_max_batch_size() const = 0;

		// The method should add testing result to the training history of each element
		// Size of random_uniform_list is a power of 2
		virtual void train_step(
			supervised_data_reader& reader,
			std::vector<training_task_state>& task_list) = 0;

		network_schema_smart_ptr schema;

	private:
		bool is_last_iteration(const training_task_state& state) const;

		bool is_broken(const training_task_state& state) const;

	private:
		network_trainer(const network_trainer&);
		network_trainer& operator =(const network_trainer&);
	};

	typedef std::tr1::shared_ptr<network_trainer> network_trainer_smart_ptr;
}
