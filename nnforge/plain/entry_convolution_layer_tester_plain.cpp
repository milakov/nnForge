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

#include "entry_convolution_layer_tester_plain.h"

#include "../entry_convolution_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		entry_convolution_layer_tester_plain::entry_convolution_layer_tester_plain()
		{
		}

		entry_convolution_layer_tester_plain::~entry_convolution_layer_tester_plain()
		{
		}

		std::string entry_convolution_layer_tester_plain::get_type_name() const
		{
			return entry_convolution_layer::layer_type_name;
		}

		void entry_convolution_layer_tester_plain::run_forward_propagation(
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
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const entry_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const entry_convolution_layer>(layer_schema);
			const unsigned int padding = layer_derived->padding;
			const int total_workload = entry_count * neuron_count_per_feature_map;
			const int input_feature_map_count = input_configuration_specific_list[0].feature_map_count;
			const int output_feature_map_count = output_configuration_specific.feature_map_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / neuron_count_per_feature_map;
					int neuron_id = workload_id - entry_id * neuron_count_per_feature_map;

					const float * in_it_base1 = in_it_global + entry_id * 2 * input_neuron_count + neuron_id;
					const float * in_it_base2 = in_it_base1 + input_neuron_count;
					float * out_it_base = out_it_global + entry_id * output_neuron_count + neuron_id;

					for(int output_index = 0; output_index < output_feature_map_count; ++output_index)
					{
						int base_input_index1 = 0;
						int base_input_index2 = output_index;
						if (output_index > (input_feature_map_count - 1))
						{
							base_input_index1 = output_index - (input_feature_map_count - 1);
							base_input_index2 = (input_feature_map_count - 1);
						}
						int iteration_count = std::min(input_feature_map_count - base_input_index1, base_input_index2 + 1);

						float sum = 0.0F;
						for(int i = 0; i < iteration_count; ++i)
							sum += in_it_base1[(base_input_index1 + i) * neuron_count_per_feature_map] * in_it_base2[(base_input_index2 - i) * neuron_count_per_feature_map];

						out_it_base[output_index * neuron_count_per_feature_map] = sum;
					}
				}
			}
		}
	}
}
