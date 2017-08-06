/*
 *  Copyright 2011-2017 Maxim Milakov
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
#include "structured_data_bunch_reader.h"
#include "training_momentum.h"
#include "learning_rate_decay_policy.h"

#include <map>
#include <memory>

namespace nnforge
{
	class network_trainer
	{
	public:
		typedef std::shared_ptr<network_trainer> ptr;

		virtual ~network_trainer() = default;

		void train(
			structured_data_bunch_reader& reader,
			network_data_peeker& peeker,
			network_data_pusher& progress_pusher,
			network_data_pusher& pusher);

		unsigned int epoch_count;
		unsigned int batch_size;
		unsigned int max_chunk_size;
		float learning_rate;
		learning_rate_decay_policy::const_ptr lr_policy;
		float weight_decay;
		training_momentum momentum;

	protected:
		network_trainer(
			network_schema::ptr schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names);

		float get_global_learning_rate(unsigned int epoch) const;

		virtual void initialize_train(structured_data_bunch_reader& reader) = 0;

		// The method should add testing result to the training history of each element
		virtual void train_step(
			structured_data_bunch_reader& reader,
			training_task_state& task) = 0;

		network_schema::ptr schema;
		std::vector<std::string> output_layer_names;
		std::vector<std::string> error_source_layer_names;
		std::vector<std::string> exclude_data_update_layer_names;

	private:
		bool is_last_epoch(const training_task_state& state) const;

		bool is_broken(const training_task_state& state) const;

	private:
		network_trainer(const network_trainer&);
		network_trainer& operator =(const network_trainer&);
	};
}
