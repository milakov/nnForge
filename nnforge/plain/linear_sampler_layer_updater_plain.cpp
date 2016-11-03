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

#include "linear_sampler_layer_updater_plain.h"

#include "../linear_sampler_layer.h"
#include "../neural_network_exception.h"

#include <cstring>

namespace nnforge
{
	namespace plain
	{
		std::string linear_sampler_layer_updater_plain::get_type_name() const
		{
			return linear_sampler_layer::layer_type_name;
		}

		void linear_sampler_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_grid_global = *input_buffers[0];
			const float * const in_it_data_global = *input_buffers[1];
			float * const out_it_global = *output_buffer;
			const unsigned int output_width = output_configuration_specific.dimension_sizes[0];
			const unsigned int output_height = output_configuration_specific.dimension_sizes[1];
			const unsigned int input_width = input_configuration_specific_list[1].dimension_sizes[0];
			const unsigned int input_height = input_configuration_specific_list[1].dimension_sizes[1];
			const unsigned int input_feature_map_count = input_configuration_specific_list[1].feature_map_count;
			const float denormalize_scale_x = static_cast<float>(input_configuration_specific_list[1].dimension_sizes[0] - 1);
			const float denormalize_scale_y = static_cast<float>(input_configuration_specific_list[1].dimension_sizes[1] - 1);
			const unsigned int output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int input_elem_count_per_feature_map = input_configuration_specific_list[1].get_neuron_count_per_feature_map();
			const unsigned int output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			const unsigned int input_elem_count_per_entry = input_configuration_specific_list[1].get_neuron_count();

			const int total_workload = entry_count * output_height * output_width;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / (output_height * output_width);
					int yx = workload_id - (entry_id * (output_height * output_width));
					int y = yx / output_width;
					int x = yx - y * output_width;

					int grid_x_offset = entry_id * output_elem_count_per_feature_map * 2 + y * output_width + x;
					int grid_y_offset = grid_x_offset + output_elem_count_per_feature_map;
					float normalized_x_pos = in_it_grid_global[grid_x_offset];
					float normalized_y_pos = in_it_grid_global[grid_y_offset];
					float absolute_x_pos = normalized_x_pos * denormalize_scale_x;
					float absolute_y_pos = normalized_y_pos * denormalize_scale_y;
					int left_x = static_cast<int>(absolute_x_pos);
					int top_y = static_cast<int>(absolute_y_pos);
					int right_x = left_x + 1;
					int bottom_y = top_y + 1;
					float right_weight = absolute_x_pos - (float)left_x;
					float left_weight = 1.0F - right_weight;
					float bottom_weight = absolute_y_pos - (float)top_y;
					float top_weight = 1.0F - bottom_weight;
					float top_left_weight = top_weight * left_weight;
					float top_right_weight = top_weight * right_weight;
					float bottom_left_weight = bottom_weight * left_weight;
					float bottom_right_weight = bottom_weight * right_weight;
					bool left_in_bounds = (unsigned int)left_x < (unsigned int)input_width;
					bool right_in_bounds = (unsigned int)right_x < (unsigned int)input_width;
					bool top_in_bounds = (unsigned int)top_y < (unsigned int)input_height;
					bool bottom_in_bounds = (unsigned int)bottom_y < (unsigned int)input_height;
					bool top_left_in_bounds = left_in_bounds && top_in_bounds;
					bool top_right_in_bounds = right_in_bounds && top_in_bounds;
					bool bottom_left_in_bounds = left_in_bounds && bottom_in_bounds;
					bool bottom_right_in_bounds = right_in_bounds && bottom_in_bounds;
					const float * current_input_data = in_it_data_global + entry_id * input_elem_count_per_entry + top_y * input_width + left_x;
					float * current_output = out_it_global + entry_id * output_elem_count_per_entry + y * output_width + x;
					for(int input_feature_map_id = 0; input_feature_map_id < static_cast<int>(input_feature_map_count); ++input_feature_map_id)
					{
						float top_left_val = top_left_in_bounds ? *current_input_data : 0.0F;
						float top_right_val = top_right_in_bounds ? *(current_input_data + 1) : 0.0F;
						float bottom_left_val = bottom_left_in_bounds ? *(current_input_data + input_width) : 0.0F;
						float bottom_right_val = bottom_right_in_bounds ? *(current_input_data + input_width + 1) : 0.0F;

						float weighted_sum = top_left_weight * top_left_val + top_right_weight * top_right_val + bottom_left_weight * bottom_left_val + bottom_right_weight * bottom_right_val;
						*current_output = weighted_sum;

						current_input_data += input_elem_count_per_feature_map;
						current_output += output_elem_count_per_entry;
					}
				}
			}
		}

		void linear_sampler_layer_updater_plain::run_backward_data_propagation(
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
			if (input_index == 1)
				throw neural_network_exception("linear_sampler_layer_updater_cuda cannot do backward propagation for input neurons");

			float * const in_it_grid_errors = *input_errors_buffer;
			const float * const in_it_grid_global = *input_neurons_buffers[0];
			const float * const in_it_data_global = *input_neurons_buffers[1];
			const float * const out_it_err_global = *output_errors_buffer;
			const unsigned int output_width = output_configuration_specific.dimension_sizes[0];
			const unsigned int output_height = output_configuration_specific.dimension_sizes[1];
			const unsigned int input_width = input_configuration_specific_list[1].dimension_sizes[0];
			const unsigned int input_height = input_configuration_specific_list[1].dimension_sizes[1];
			const unsigned int input_feature_map_count = input_configuration_specific_list[1].feature_map_count;
			const float denormalize_scale_x = static_cast<float>(input_configuration_specific_list[1].dimension_sizes[0] - 1);
			const float denormalize_scale_y = static_cast<float>(input_configuration_specific_list[1].dimension_sizes[1] - 1);
			const unsigned int output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int input_elem_count_per_feature_map = input_configuration_specific_list[1].get_neuron_count_per_feature_map();
			const unsigned int output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			const unsigned int input_elem_count_per_entry = input_configuration_specific_list[1].get_neuron_count();

			const int total_workload = entry_count * output_height * output_width;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / (output_height * output_width);
					int yx = workload_id - (entry_id * (output_height * output_width));
					int y = yx / output_width;
					int x = yx - y * output_width;

					int grid_x_offset = entry_id * output_elem_count_per_feature_map * 2 + y * output_width + x;
					int grid_y_offset = grid_x_offset + output_elem_count_per_feature_map;
					float normalized_x_pos = in_it_grid_global[grid_x_offset];
					float normalized_y_pos = in_it_grid_global[grid_y_offset];
					float absolute_x_pos = normalized_x_pos * denormalize_scale_x;
					float absolute_y_pos = normalized_y_pos * denormalize_scale_y;
					int left_x = static_cast<int>(absolute_x_pos);
					int top_y = static_cast<int>(absolute_y_pos);
					int right_x = left_x + 1;
					int bottom_y = top_y + 1;
					bool left_in_bounds = (unsigned int)left_x < (unsigned int)input_width;
					bool right_in_bounds = (unsigned int)right_x < (unsigned int)input_width;
					bool top_in_bounds = (unsigned int)top_y < (unsigned int)input_height;
					bool bottom_in_bounds = (unsigned int)bottom_y < (unsigned int)input_height;
					bool top_left_in_bounds = left_in_bounds && top_in_bounds;
					bool top_right_in_bounds = right_in_bounds && top_in_bounds;
					bool bottom_left_in_bounds = left_in_bounds && bottom_in_bounds;
					bool bottom_right_in_bounds = right_in_bounds && bottom_in_bounds;
					const float * current_input_data = in_it_data_global + entry_id * input_elem_count_per_entry + top_y * input_width + left_x;
					const float * current_output_errors = out_it_err_global + entry_id * output_elem_count_per_entry + y * output_width + x;
					float top_left_sum = 0.0F;
					float top_right_sum = 0.0F;
					float bottom_left_sum = 0.0F;
					float bottom_right_sum = 0.0F;
					for(int input_feature_map_id = 0; input_feature_map_id < static_cast<int>(input_feature_map_count); ++input_feature_map_id)
					{
						float output_error = *current_output_errors;

						float top_left_val = top_left_in_bounds ? *current_input_data : 0.0F;
						float top_right_val = top_right_in_bounds ? *(current_input_data + 1) : 0.0F;
						float bottom_left_val = bottom_left_in_bounds ? *(current_input_data + input_width) : 0.0F;
						float bottom_right_val = bottom_right_in_bounds ? *(current_input_data + input_width + 1) : 0.0F;

						top_left_sum += top_left_val * output_error;
						top_right_sum += top_right_val * output_error;
						bottom_left_sum += bottom_left_val * output_error;
						bottom_right_sum += bottom_right_val * output_error;

						current_input_data += input_elem_count_per_feature_map;
						current_output_errors += output_elem_count_per_entry;
					}

					float right_weight = absolute_x_pos - (float)left_x;
					float left_weight = 1.0F - right_weight;
					float bottom_weight = absolute_y_pos - (float)top_y;
					float top_weight = 1.0F - bottom_weight;

					float input_err_x = (top_weight * (top_right_sum - top_left_sum) + bottom_weight * (bottom_right_sum - bottom_left_sum)) * denormalize_scale_x;
					float input_err_y = (left_weight * (bottom_left_sum - top_left_sum) + right_weight * (bottom_right_sum - top_right_sum)) * denormalize_scale_y;

					if (add_update_to_destination)
					{
						in_it_grid_errors[grid_x_offset] += input_err_x;
						in_it_grid_errors[grid_y_offset] += input_err_y;
					}
					else
					{
						in_it_grid_errors[grid_x_offset] = input_err_x;
						in_it_grid_errors[grid_y_offset] = input_err_y;
					}
				}
			}
		}

		bool linear_sampler_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
			unsigned int action_input_index,
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return (action_input_index == 0);
		}

		bool linear_sampler_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
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
