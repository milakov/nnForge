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

#include "mse_layer_updater_plain.h"

#include "../mse_layer.h"

namespace nnforge
{
	namespace plain
	{
		mse_layer_updater_plain::mse_layer_updater_plain()
		{
		}

		mse_layer_updater_plain::~mse_layer_updater_plain()
		{
		}

		std::string mse_layer_updater_plain::get_type_name() const
		{
			return mse_layer::layer_type_name;
		}

		void mse_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_global0 = *input_buffers[0];
			const float * const in_it_global1 = *input_buffers[1];
			float * const out_it_global = *output_buffer;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const int input_feature_map_count = static_cast<int>(input_configuration_specific_list[0].feature_map_count);
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			nnforge_shared_ptr<const mse_layer> layer_derived = nnforge_dynamic_pointer_cast<const mse_layer>(layer_schema);
			const float scale = layer_derived->scale;
			const int total_workload = entry_count * output_neuron_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_neuron_count;
					int output_neuron_id = workload_id - (entry_id * output_neuron_count);

					const float * in_it_base0 = in_it_global0 + entry_id * input_neuron_count + output_neuron_id;
					const float * in_it_base1 = in_it_global1 + entry_id * input_neuron_count + output_neuron_id;
					float * out_it = out_it_global + entry_id * output_neuron_count + output_neuron_id;

					float err = 0.0F;
					for(int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
					{
						float local_err = *(in_it_base0 + feature_map_id * input_neuron_count_per_feature_map) - *(in_it_base1 + feature_map_id * input_neuron_count_per_feature_map);
						err += local_err * local_err;
					}

					*out_it = err * scale;
				}
			}
		}

		void mse_layer_updater_plain::run_backward_data_propagation(
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
			float * const in_err_it = *input_errors_buffer;
			const float * const deriv_input_neurons_it = *input_neurons_buffers[input_index];
			const float * const target_input_neurons_it = *input_neurons_buffers[1 - input_index];

			nnforge_shared_ptr<const mse_layer> layer_derived = nnforge_dynamic_pointer_cast<const mse_layer>(layer_schema);
			const float scale2 = layer_derived->scale * 2.0F;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();

			const int total_workload = entry_count * input_neuron_count;
			if (add_update_to_destination)
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < total_workload; ++i)
				{
					*(in_err_it + i) += scale2 * (*(target_input_neurons_it + i) - *(deriv_input_neurons_it + i));
				}
			}
			else
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < total_workload; ++i)
				{
					*(in_err_it + i) = scale2 * (*(target_input_neurons_it + i) - *(deriv_input_neurons_it + i));
				}
			}
		}

		bool mse_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
			unsigned int action_input_index,
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return true;
		}

		bool mse_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}
	}
}
