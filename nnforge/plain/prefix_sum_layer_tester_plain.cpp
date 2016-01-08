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

#include "prefix_sum_layer_tester_plain.h"

#include "../prefix_sum_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		prefix_sum_layer_tester_plain::prefix_sum_layer_tester_plain()
		{
		}

		prefix_sum_layer_tester_plain::~prefix_sum_layer_tester_plain()
		{
		}

		std::string prefix_sum_layer_tester_plain::get_type_name() const
		{
			return prefix_sum_layer::layer_type_name;
		}

		void prefix_sum_layer_tester_plain::run_forward_propagation(
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
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			nnforge_shared_ptr<const prefix_sum_layer> layer_derived = nnforge_dynamic_pointer_cast<const prefix_sum_layer>(layer_schema);
			const unsigned int feature_map_segment_length = layer_derived->feature_map_segment_length;
			const unsigned int feature_map_segment_count = output_configuration_specific.feature_map_count / feature_map_segment_length;
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const float clamp_min = layer_derived->clamp_min;
			const float clamp_max = layer_derived->clamp_max;
			const int total_workload = entry_count * feature_map_segment_count * neuron_count_per_feature_map;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / (feature_map_segment_count * neuron_count_per_feature_map);
					int tt = workload_id - entry_id * feature_map_segment_count * neuron_count_per_feature_map;
					int feature_map_segment_id = tt / neuron_count_per_feature_map;
					int neuron_id = tt - feature_map_segment_id * neuron_count_per_feature_map;

					int offset = entry_id * neuron_count + feature_map_segment_id * feature_map_segment_length * neuron_count_per_feature_map + neuron_id;

					float running_sum = 0.0F;
					for(unsigned int i = 0; i < feature_map_segment_length; ++i, offset += neuron_count_per_feature_map)
					{
						running_sum += in_it_global[offset];
						out_it_global[offset] = std::min(std::max(running_sum, clamp_min), clamp_max);
					}
				}
			}
		}

		int prefix_sum_layer_tester_plain::get_input_index_layer_can_write(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}
	}
}
