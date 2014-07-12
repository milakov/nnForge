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

#include "softmax_layer_updater_plain.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../softmax_layer.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		softmax_layer_updater_plain::softmax_layer_updater_plain()
		{
		}

		softmax_layer_updater_plain::~softmax_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& softmax_layer_updater_plain::get_uuid() const
		{
			return softmax_layer::layer_guid;
		}

		void softmax_layer_updater_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id) const
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("softmax_layer_updater_plain is not able to run using offset");

			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = static_cast<unsigned int>(input_configuration_specific.feature_map_count);

			const std::vector<float>::const_iterator input_buffer_it = input_buffer->begin();
			const std::vector<float>::iterator output_buffer_it = output_buffer->begin();

			const int total_workload = updater_count * input_neuron_count_per_feature_map;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) shared(additional_buffers) num_threads(openmp_thread_count)
			{
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				std::vector<float>& local_additional_buffer = *(additional_buffers[thread_id]);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / input_neuron_count_per_feature_map;
					int neuron_id = workload_id - (entry_id * input_neuron_count_per_feature_map);

					const std::vector<float>::const_iterator in_it = input_buffer_it + (entry_id * input_neuron_count) + neuron_id;
					const std::vector<float>::iterator out_it = output_buffer_it + (entry_id * input_neuron_count) + neuron_id;

					float sum = 0.0F;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						float val = expf(*(in_it + (feature_map_id * input_neuron_count_per_feature_map)));
						sum += val;
						local_additional_buffer[feature_map_id] = val;
					}
					float mult = 1.0F / sum;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
						*(out_it + (feature_map_id * input_neuron_count_per_feature_map)) = local_additional_buffer[feature_map_id] * mult;
				} // for(int workload_id
			} // #pragma parallel
		}

		void softmax_layer_updater_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = static_cast<unsigned int>(input_configuration_specific.feature_map_count);

			const std::vector<float>::iterator input_errors_it = input_errors->begin();
			const std::vector<float>::const_iterator output_errors_it = output_errors->begin();
			const std::vector<float>::const_iterator output_neurons_it = output_neurons->begin();

			const int total_workload = updater_count * input_neuron_count_per_feature_map;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) shared(additional_buffers) num_threads(openmp_thread_count)
			{
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / input_neuron_count_per_feature_map;
					int neuron_id = workload_id - (entry_id * input_neuron_count_per_feature_map);

					const std::vector<float>::iterator in_errors_it = input_errors_it + (entry_id * input_neuron_count) + neuron_id;
					const std::vector<float>::const_iterator out_errors_it = output_errors_it + (entry_id * input_neuron_count) + neuron_id;
					const std::vector<float>::const_iterator out_neurons_it = output_neurons_it + (entry_id * input_neuron_count) + neuron_id;

					float sum = 0.0F;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						unsigned int offset = feature_map_id * input_neuron_count_per_feature_map;
						sum += (*(out_errors_it + offset)) * (*(out_neurons_it + offset));
					}

					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						unsigned int offset = feature_map_id * input_neuron_count_per_feature_map;
						*(in_errors_it + offset) = (*(out_neurons_it + offset)) * (*(out_errors_it + offset) - sum);
					}
				} // for(int workload_id
			} // #pragma parallel
		}

		bool softmax_layer_updater_plain::is_in_place_backprop() const
		{
			return true;
		}

		std::vector<std::pair<unsigned int, bool> > softmax_layer_updater_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			for(int i = 0; i < plain_config->openmp_thread_count; ++i)
				res.push_back(std::make_pair(input_configuration_specific.feature_map_count, false));

			return res;
		}
	}
}
