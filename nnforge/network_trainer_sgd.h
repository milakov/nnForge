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

#include "network_trainer.h"
#include "backward_propagation.h"

#include <vector>

namespace nnforge
{
	// Stochastic Gradient Descent
	class network_trainer_sgd : public network_trainer
	{
	public:
		typedef nnforge_shared_ptr<network_trainer_sgd> ptr;

		network_trainer_sgd(
			network_schema::ptr schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names,
			backward_propagation::ptr backprop);

		virtual ~network_trainer_sgd();

	protected:
		// The method should add testing result to the training history of each element
		virtual void train_step(
			structured_data_bunch_reader& reader,
			training_task_state& task);

		virtual void initialize_train(structured_data_bunch_reader& reader);

	private:
		std::pair<std::map<std::string, std::vector<float> >, std::string> prepare_learning_rates(
			unsigned int epoch,
			network_data::const_ptr data);

	private:
		backward_propagation::ptr backprop;
	};
}
