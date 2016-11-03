/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "parametric_rectified_linear_layer_updater_plain.h"

#include "../parametric_rectified_linear_layer.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		std::string parametric_rectified_linear_layer_updater_plain::get_type_name() const
		{
			return parametric_rectified_linear_layer::layer_type_name;
		}

		void parametric_rectified_linear_layer_updater_plain::run_forward_propagation(
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
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;

			const int total_workload = static_cast<int>(entry_count * feature_map_count);
			const float * const in_it = *input_buffers[0];
			float * const out_it = *output_buffer;
			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float a = weights[feature_map_id];

				const float * current_in_it = in_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
				float * current_out_it = out_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);

				for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				{
					float input_val = *(current_in_it + i);
					float output_val = input_val * (input_val >= 0.0F ? 1.0F : a);
					*(current_out_it + i) = output_val;
				}
			}
		}

		void parametric_rectified_linear_layer_updater_plain::run_backward_data_propagation(
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
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;

			const int total_workload = static_cast<int>(entry_count * feature_map_count);
			const float * const in_neurons_it = *input_neurons_buffers[0];
			const float * const out_errors_it = *output_errors_buffer;
			float * const  in_errors_it = *input_errors_buffer;
			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float a = weights[feature_map_id];

				const float * current_in_neurons_it = in_neurons_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
				float * current_in_errors_it = in_errors_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
				const float * current_out_errors_it = out_errors_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);

				if (add_update_to_destination)
				{
					for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
					{
						float output_err = *(current_out_errors_it+ i);
						float input_val = *(current_in_neurons_it + i);
						float input_err = output_err * (input_val >= 0.0F ? 1.0F : a);
						*(current_in_errors_it + i) += input_err;
					}
				}
				else
				{
					for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
					{
						float output_err = *(current_out_errors_it+ i);
						float input_val = *(current_in_neurons_it + i);
						float input_err = output_err * (input_val >= 0.0F ? 1.0F : a);
						*(current_in_errors_it + i) = input_err;
					}
				}
			}
		}

		void parametric_rectified_linear_layer_updater_plain::run_backward_weights_propagation(
			const std::vector<plain_buffer::const_ptr>& input_neurons_buffers,
			plain_buffer::const_ptr output_errors_buffer,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_buffer::ptr temporary_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::ptr gradient,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			const std::set<layer_action>& actions,
			unsigned int entry_count) const
		{
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;

			const float * const in_neurons_it = *input_neurons_buffers[0];
			const float * const err_it = *output_errors_buffer;
			const std::vector<float>::iterator gradients = (*gradient)[0].begin();

			const int total_workload = feature_map_count;
			const int const_updater_count = entry_count;

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int feature_map_id = workload_id;

				float sum = 0.0F;
				for(int entry_id = 0; entry_id < const_updater_count; ++entry_id)
				{
					const float * current_in_neurons_it = in_neurons_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
					const float * current_err_it = err_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);

					float local_sum = 0.0F;
					for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
					{
						float output_err = *(current_err_it + i);
						float input_val = *(current_in_neurons_it + i);
						float gr = output_err * (input_val >= 0.0F ? 0.0F : input_val);
						local_sum += gr;
					}

					sum += local_sum;
				}

				*(gradients + feature_map_id) += sum;
			}
		}

		int parametric_rectified_linear_layer_updater_plain::get_input_index_layer_can_write(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}

		bool parametric_rectified_linear_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool parametric_rectified_linear_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		bool parametric_rectified_linear_layer_updater_plain::is_backward_weights_dependent_on_input_buffer(
			unsigned int data_input_index,
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
