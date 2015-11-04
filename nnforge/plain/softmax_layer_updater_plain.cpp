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

#include "softmax_layer_updater_plain.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../softmax_layer.h"

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

		std::string softmax_layer_updater_plain::get_type_name() const
		{
			return softmax_layer::layer_type_name;
		}

		void softmax_layer_updater_plain::run_forward_propagation(
			plain_buffer::ptr output_buffer,
			const std::vector<plain_buffer::const_ptr>& input_buffers,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_buffer::ptr temporary_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::const_ptr data,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			const std::set<layer_action>& actions,
			unsigned int entry_count) const
		{
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = static_cast<unsigned int>(output_configuration_specific.feature_map_count);

			const float * const input_buffer_it = *input_buffers[0];
			float * const output_buffer_it = *output_buffer;
			float * const working_buffer_it = *temporary_working_fixed_buffer;

			const int total_workload = entry_count * neuron_count_per_feature_map;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) num_threads(openmp_thread_count)
			{
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				float * local_additional_buffer = working_buffer_it + thread_id * feature_map_count;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / neuron_count_per_feature_map;
					int neuron_id = workload_id - (entry_id * neuron_count_per_feature_map);

					const float * in_it = input_buffer_it + (entry_id * neuron_count) + neuron_id;
					float * out_it = output_buffer_it + (entry_id * neuron_count) + neuron_id;

					float max_val = -1.0e+37F;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						float val = *(in_it + (feature_map_id * neuron_count_per_feature_map));
						max_val = std::max(max_val, val);
					}

					float sum = 0.0F;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						float val = expf((*(in_it + (feature_map_id * neuron_count_per_feature_map))) - max_val);
						sum += val;
						local_additional_buffer[feature_map_id] = val;
					}
					float mult = 1.0F / sum;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
						*(out_it + (feature_map_id * neuron_count_per_feature_map)) = local_additional_buffer[feature_map_id] * mult;
				} // for(int workload_id
			} // #pragma parallel
		}

		void softmax_layer_updater_plain::run_backward_data_propagation(
			unsigned int input_index,
			plain_buffer::ptr input_errors_buffer,
			plain_buffer::const_ptr output_errors_buffer,
			const std::vector<plain_buffer::const_ptr>& input_neurons_buffers,
			plain_buffer::const_ptr output_neurons_buffer,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_buffer::ptr temporary_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::const_ptr data,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			const bool add_update_to_destination,
			const std::set<layer_action>& actions,
			unsigned int entry_count) const
		{
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = static_cast<unsigned int>(output_configuration_specific.feature_map_count);

			float * const input_errors_it = *input_errors_buffer;
			const float * const output_errors_it = *output_errors_buffer;
			const float * const output_neurons_it = *output_neurons_buffer;

			const int total_workload = entry_count * neuron_count_per_feature_map;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) num_threads(openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / neuron_count_per_feature_map;
					int neuron_id = workload_id - (entry_id * neuron_count_per_feature_map);

					float * in_errors_it = input_errors_it + (entry_id * neuron_count) + neuron_id;
					const float * out_errors_it = output_errors_it + (entry_id * neuron_count) + neuron_id;
					const float * out_neurons_it = output_neurons_it + (entry_id * neuron_count) + neuron_id;

					float sum = 0.0F;
					for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
					{
						unsigned int offset = feature_map_id * neuron_count_per_feature_map;
						sum += (*(out_errors_it + offset)) * (*(out_neurons_it + offset));
					}

					if (add_update_to_destination)
					{
						for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
						{
							unsigned int offset = feature_map_id * neuron_count_per_feature_map;
							*(in_errors_it + offset) += (*(out_neurons_it + offset)) * (*(out_errors_it + offset) - sum);
						}
					}
					else
					{
						for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
						{
							unsigned int offset = feature_map_id * neuron_count_per_feature_map;
							*(in_errors_it + offset) = (*(out_neurons_it + offset)) * (*(out_errors_it + offset) - sum);
						}
					}
				} // for(int workload_id
			} // #pragma parallel
		}

		int softmax_layer_updater_plain::get_input_index_layer_can_write(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}

		size_t softmax_layer_updater_plain::get_temporary_working_fixed_buffer_size(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (action.get_action_type() == layer_action::forward)
			{
				return plain_config->openmp_thread_count * output_configuration_specific.feature_map_count * sizeof(float);
			}
			else
				return 0;
		}

		bool softmax_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
			unsigned int action_input_index,
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		bool softmax_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return true;
		}
	}
}
