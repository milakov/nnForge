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

#include "negative_log_likelihood_layer_tester_plain.h"

#include "../negative_log_likelihood_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		negative_log_likelihood_layer_tester_plain::negative_log_likelihood_layer_tester_plain()
		{
		}

		negative_log_likelihood_layer_tester_plain::~negative_log_likelihood_layer_tester_plain()
		{
		}

		std::string negative_log_likelihood_layer_tester_plain::get_type_name() const
		{
			return negative_log_likelihood_layer::layer_type_name;
		}

		void negative_log_likelihood_layer_tester_plain::run_forward_propagation(
			plain_buffer::ptr output_buffer,
			const std::vector<plain_buffer::const_ptr>& input_buffers,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::const_ptr data,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
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
			nnforge_shared_ptr<const negative_log_likelihood_layer> layer_derived = nnforge_dynamic_pointer_cast<const negative_log_likelihood_layer>(layer_schema);
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
								err -= actual_val * logf(std::max(predicted_val, 1.0e-20F));
						}
						err *= total_scale;
					}

					*(out_it_global + output_offset) = err;
				}
			}
		}
	}
}
