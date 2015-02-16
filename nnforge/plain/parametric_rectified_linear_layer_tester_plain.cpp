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

#include "parametric_rectified_linear_layer_tester_plain.h"

#include "../parametric_rectified_linear_layer.h"

namespace nnforge
{
	namespace plain
	{
		parametric_rectified_linear_layer_tester_plain::parametric_rectified_linear_layer_tester_plain()
		{
		}

		parametric_rectified_linear_layer_tester_plain::~parametric_rectified_linear_layer_tester_plain()
		{
		}

		const boost::uuids::uuid& parametric_rectified_linear_layer_tester_plain::get_uuid() const
		{
			return parametric_rectified_linear_layer::layer_guid;
		}

		void parametric_rectified_linear_layer_tester_plain::test(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const int total_workload = static_cast<int>(entry_count * input_configuration_specific.feature_map_count);
			const std::vector<float>::iterator in_it = input_buffer->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = input_configuration_specific.feature_map_count;
			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float a = weights[feature_map_id];

				std::vector<float>::iterator current_it = in_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
				std::vector<float>::const_iterator current_it_end = current_it + input_neuron_count_per_feature_map;

				for(; current_it != current_it_end; ++current_it)
				{
					float input_val = *current_it;
					float output_val = input_val * (input_val >= 0.0F ? 1.0F : a);
					*current_it = output_val;
				}
			}
		}
	}
}
