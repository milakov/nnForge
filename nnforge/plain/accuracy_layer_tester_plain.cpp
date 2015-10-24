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

#include "accuracy_layer_tester_plain.h"

#include "../accuracy_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		accuracy_layer_tester_plain::accuracy_layer_tester_plain()
		{
		}

		accuracy_layer_tester_plain::~accuracy_layer_tester_plain()
		{
		}

		std::string accuracy_layer_tester_plain::get_type_name() const
		{
			return accuracy_layer::layer_type_name;
		}

		void accuracy_layer_tester_plain::run_forward_propagation(
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
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const int input_feature_map_count = static_cast<int>(input_configuration_specific_list[0].feature_map_count);
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const int output_feature_map_count = static_cast<int>(output_configuration_specific.feature_map_count);
			const int total_workload = entry_count * output_neuron_count_per_feature_map;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_neuron_count_per_feature_map;
					int output_neuron_id = workload_id - (entry_id * output_neuron_count_per_feature_map);

					const float * in_it_base_predicted = in_it_global_predicted + entry_id * input_neuron_count + output_neuron_id;
					const float * in_it_base_actual = in_it_global_actual + entry_id * input_neuron_count + output_neuron_id;
					float * out_it = out_it_global + entry_id * output_neuron_count + output_neuron_id;

					float max_val = -1.0e37F;
					int max_val_feature_map_id = -1;
					for(int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
					{
						float actual_val = *(in_it_base_actual + feature_map_id * input_neuron_count_per_feature_map);
						if (actual_val > max_val)
						{
							max_val = actual_val;
							max_val_feature_map_id = feature_map_id;
						}
					}
					// max_val_feature_map_id identifies actual class

					max_val = *(in_it_base_predicted + max_val_feature_map_id * input_neuron_count_per_feature_map);
					// max_val_feature_map_id identifies actual class
					// max_val is equal to the value predicted for that class

					int sum = 0;
					for(int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
					{
						float val = *(in_it_base_predicted + feature_map_id * input_neuron_count_per_feature_map);
						if ((val > max_val) || ((val == max_val) && (feature_map_id < max_val_feature_map_id)))
							++sum;
					}

					for(int output_feature_map_id = 0; output_feature_map_id < output_feature_map_count; ++output_feature_map_id)
					{
						*(out_it + output_feature_map_id * output_neuron_count_per_feature_map) = ((sum <= output_feature_map_id) ? 1.0F : 0.0F);
					}
				}
			}
		}
	}
}
