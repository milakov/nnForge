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

#include "maxout_layer_updater_plain.h"

#include "../maxout_layer.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		std::string maxout_layer_updater_plain::get_type_name() const
		{
			return maxout_layer::layer_type_name;
		}

		void maxout_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			unsigned int * const max_feature_map_positions_it_global = *temporary_per_entry_buffer;

			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			std::shared_ptr<const maxout_layer> layer_derived = std::dynamic_pointer_cast<const maxout_layer>(layer_schema);
			const unsigned int feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
			const int output_feature_map_count = output_configuration_specific.feature_map_count;
			const int total_workload = entry_count * output_feature_map_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					const float * in_it_base = in_it_global + (entry_id * input_neuron_count) + (output_feature_map_id * input_neuron_count_per_feature_map);
					int output_offset = (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					float * out_it_base = out_it_global + output_offset;
					unsigned int * max_feature_map_positions_it = max_feature_map_positions_it_global + output_offset;

					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++max_feature_map_positions_it, ++in_it_base)
					{
						const float * in_it = in_it_base;
						float current_max = *in_it;
						int max_feature_map_pos = 0;
						for(unsigned int i = 1; i < feature_map_subsampling_size; ++i)
						{
							in_it += output_feature_map_count * output_neuron_count_per_feature_map;
							float new_val = *in_it;
							if (new_val > current_max)
							{
								current_max = new_val;
								max_feature_map_pos = i;
							}
						}
						*out_it = current_max;
						*max_feature_map_positions_it = max_feature_map_pos;
					}
				}
			}
		}

		void maxout_layer_updater_plain::run_backward_data_propagation(
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
			float * const in_err_it_global = *input_errors_buffer;
			const float * const out_err_it_global = *output_errors_buffer;
			const unsigned int * const max_feature_map_positions_it_global = *temporary_per_entry_buffer;

			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			std::shared_ptr<const maxout_layer> layer_derived = std::dynamic_pointer_cast<const maxout_layer>(layer_schema);
			const unsigned int feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
			const int output_feature_map_count = output_configuration_specific.feature_map_count;
			const int total_workload = entry_count * output_feature_map_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					float * in_err_it_base = in_err_it_global + (entry_id * input_neuron_count) + (output_feature_map_id * input_neuron_count_per_feature_map);
					int output_offset = (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					const float * out_err_it_base = out_err_it_global + output_offset;
					const unsigned int * max_feature_map_positions_it = max_feature_map_positions_it_global + output_offset;

					if (add_update_to_destination)
					{
						for(const float * out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it, ++max_feature_map_positions_it, ++in_err_it_base)
						{
							float * in_err_it = in_err_it_base;
							float current_err = *out_err_it;
							unsigned int max_feature_map_position = *((const unsigned int *)(&(*max_feature_map_positions_it)));
							for(unsigned int i = 0; i < feature_map_subsampling_size; ++i)
								in_err_it[output_feature_map_count * output_neuron_count_per_feature_map * i] += ((i == max_feature_map_position) ? current_err : 0.0F);
						}
					}
					else
					{
						for(const float * out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it, ++max_feature_map_positions_it, ++in_err_it_base)
						{
							float * in_err_it = in_err_it_base;
							float current_err = *out_err_it;
							unsigned int max_feature_map_position = *((const unsigned int *)(&(*max_feature_map_positions_it)));
							for(unsigned int i = 0; i < feature_map_subsampling_size; ++i)
								in_err_it[output_feature_map_count * output_neuron_count_per_feature_map * i] = ((i == max_feature_map_position) ? current_err : 0.0F);
						}
					}
				}
			}
		}

		bool maxout_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool maxout_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		size_t maxout_layer_updater_plain::get_temporary_per_entry_buffer_size(
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return output_configuration_specific.get_neuron_count() * sizeof(unsigned int);
		}
	}
}
