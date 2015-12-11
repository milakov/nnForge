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

#include "cross_entropy_layer_updater_plain.h"

#include "../cross_entropy_layer.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		cross_entropy_layer_updater_plain::cross_entropy_layer_updater_plain()
		{
		}

		cross_entropy_layer_updater_plain::~cross_entropy_layer_updater_plain()
		{
		}

		std::string cross_entropy_layer_updater_plain::get_type_name() const
		{
			return cross_entropy_layer::layer_type_name;
		}

		void cross_entropy_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_global_predicted = *input_buffers[0];
			const float * const in_it_global_actual = *input_buffers[1];
			float * const out_it_global = *output_buffer;
			const float * scale_mask_it = 0;
			if (input_buffers.size() > 2)
				scale_mask_it = *input_buffers[2];
			const float * const const_scale_mask_it = scale_mask_it;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const int input_feature_map_count = static_cast<int>(input_configuration_specific_list[0].feature_map_count);
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			nnforge_shared_ptr<const cross_entropy_layer> layer_derived = nnforge_dynamic_pointer_cast<const cross_entropy_layer>(layer_schema);
			const float scale = layer_derived->scale;
			const int total_workload = entry_count * output_neuron_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_neuron_count;
					int output_neuron_id = workload_id - (entry_id * output_neuron_count);

					const float * in_it_base_predicted = in_it_global_predicted + entry_id * input_neuron_count + output_neuron_id;
					const float * in_it_base_actual = in_it_global_actual + entry_id * input_neuron_count + output_neuron_id;
					int output_offset = entry_id * output_neuron_count + output_neuron_id;

					float total_scale = scale;
					if (const_scale_mask_it)
						total_scale *= *(const_scale_mask_it + output_offset);

					float err = 0.0F;
					if (total_scale != 0.0F)
					{
						for(int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
						{
							float predicted_val = *(in_it_base_predicted + feature_map_id * input_neuron_count_per_feature_map);
							float actual_val = *(in_it_base_actual + feature_map_id * input_neuron_count_per_feature_map);

							if (actual_val > 0.0F)
							{
								err -= actual_val * logf(std::max(predicted_val, 1.0e-20F));
							}
							if (actual_val < 1.0F)
							{
								err -= (1.0F - actual_val) * logf(std::max(1.0F - predicted_val, 1.0e-20F));
							}
						}
						err *= total_scale;
					}

					*(out_it_global + output_offset) = err;
				}
			}
		}

		void cross_entropy_layer_updater_plain::run_backward_data_propagation(
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
				throw neural_network_exception("cross_entropy_layer_updater_plain cannot do backward propagation for targets");
			if (input_index == 2)
				throw neural_network_exception("cross_entropy_layer_updater_plain cannot do backward propagation for scale mask");

			float * const in_err_it = *input_errors_buffer;
			const float * const deriv_input_neurons_it = *input_neurons_buffers[0];
			const float * const target_input_neurons_it = *input_neurons_buffers[1];
			const float * scale_mask_it = 0;
			if (input_neurons_buffers.size() > 2)
				scale_mask_it = *input_neurons_buffers[2];
			const float * const const_scale_mask_it = scale_mask_it;

			nnforge_shared_ptr<const cross_entropy_layer> layer_derived = nnforge_dynamic_pointer_cast<const cross_entropy_layer>(layer_schema);
			const float scale = layer_derived->scale;
			const int neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const int input_feature_map_count = input_configuration_specific_list[0].feature_map_count;

			const int total_workload = entry_count * neuron_count_per_feature_map;
			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / neuron_count_per_feature_map;
				int neuron_id = workload_id - (entry_id * neuron_count_per_feature_map);
				int output_offset = entry_id * neuron_count_per_feature_map + neuron_id;
				float total_scale = scale;
				if (const_scale_mask_it)
					total_scale *= *(const_scale_mask_it + output_offset);

				for(int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
				{
					float gradient = 0.0F;
					int input_offset = (entry_id * input_feature_map_count + feature_map_id) * neuron_count_per_feature_map + neuron_id;
					if (total_scale != 0.0F)
					{
						float actual_val = *(target_input_neurons_it + input_offset);
						float predicted_val = *(deriv_input_neurons_it + input_offset);
						float gradient = 0.0F;
						if (actual_val > 0.0F)
						{
							gradient = actual_val / std::max(predicted_val, 1.0e-20F);
						}
						if (actual_val < 1.0F)
						{
							gradient -= (1.0F - actual_val) / std::max(1.0F - predicted_val, 1.0e-20F);
						}
						gradient *= total_scale;
					}

					if (add_update_to_destination)
						*(in_err_it + input_offset) += gradient;
					else
						*(in_err_it + input_offset) = gradient;
				}
			}
		}

		bool cross_entropy_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool cross_entropy_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
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
