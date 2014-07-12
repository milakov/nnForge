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
#include "hessian_calculator.h"
#include "network_updater.h"

#include <vector>
#include <string>

// http://yann.lecun.com/exdb/lenet/index.html Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition"
// http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf Y. LeCun, L. Bottou, G. Orr, and K. Muller, "Efficient BackProp"
namespace nnforge
{
	// Stochastic Diagonal Levenberg Marquardt
	class network_trainer_sdlm : public network_trainer
	{
	public:
		network_trainer_sdlm(
			network_schema_smart_ptr schema,
			hessian_calculator_smart_ptr hessian_calc,
			network_updater_smart_ptr updater);

		virtual ~network_trainer_sdlm();

		float hessian_entry_to_process_ratio;
		float max_mu;
		float mu_increase_factor;
		bool per_layer_mu;

	protected:
		// The method should add testing result to the training history of each element
		virtual void train_step(
			supervised_data_reader& reader,
			training_task_state& task);

		virtual void initialize_train(supervised_data_reader& reader);

	private:
		class hessian_transform
		{
		public:
			hessian_transform(float mu, float eta);
			
			float operator() (float in);

		private:
			float mu;
			float eta;
		};

		static const unsigned int min_hessian_entry_to_process_count;

		hessian_calculator_smart_ptr hessian_calc;
		network_updater_smart_ptr updater;

		std::vector<std::vector<float> > get_average_hessian_list(network_data_smart_ptr hessian) const;

		std::string convert_hessian_to_training_vector(
			network_data_smart_ptr hessian,
			const std::vector<std::vector<float> >& average_hessian_list,
			unsigned int epoch_id) const;

		std::string convert_hessian_to_training_vector_per_layer_mu(
			network_data_smart_ptr hessian,
			const std::vector<std::vector<float> >& average_hessian_list,
			unsigned int epoch_id) const;

		std::string convert_hessian_to_training_vector(
			network_data_smart_ptr hessian,
			unsigned int epoch_id) const;

#ifdef NNFORGE_DEBUG_HESSIAN
		void dump_lists(
			network_data_smart_ptr hessian,
			const char * filename_prefix) const;
#endif
	};

	typedef nnforge_shared_ptr<network_trainer_sdlm> network_trainer_sdlm_smart_ptr;
}
