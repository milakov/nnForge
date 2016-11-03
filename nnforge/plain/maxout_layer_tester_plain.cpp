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

#include "maxout_layer_tester_plain.h"

#include "../maxout_layer.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		std::string maxout_layer_tester_plain::get_type_name() const
		{
			return maxout_layer::layer_type_name;
		}

		void maxout_layer_tester_plain::run_forward_propagation(
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
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
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
					float * out_it_base = out_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++in_it_base)
					{
						const float * in_it = in_it_base;
						float current_max = *in_it;
						for(unsigned int i = 1; i < feature_map_subsampling_size; ++i)
						{
							in_it += output_feature_map_count * output_neuron_count_per_feature_map;
							float new_val = *in_it;
							current_max = std::max(new_val, current_max);
						}
						*out_it = current_max;
					}
				}
			}
		}
	}
}
