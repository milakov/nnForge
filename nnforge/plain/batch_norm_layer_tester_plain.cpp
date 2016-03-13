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

#include "batch_norm_layer_tester_plain.h"

#include "../batch_norm_layer.h"

namespace nnforge
{
	namespace plain
	{
		batch_norm_layer_tester_plain::batch_norm_layer_tester_plain()
		{
		}

		batch_norm_layer_tester_plain::~batch_norm_layer_tester_plain()
		{
		}

		std::string batch_norm_layer_tester_plain::get_type_name() const
		{
			return batch_norm_layer::layer_type_name;
		}

		void batch_norm_layer_tester_plain::run_forward_propagation(
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
			const int total_workload = static_cast<int>(entry_count * output_configuration_specific.feature_map_count);
			float * const out_it = *output_buffer;
			const float * const in_it = *input_buffers[0];
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;
			const std::vector<float>::const_iterator gamma = (*data)[0].begin();
			const std::vector<float>::const_iterator beta = (*data)[1].begin();
			const std::vector<float>::const_iterator mean = (*data)[2].begin();
			const std::vector<float>::const_iterator invvar = (*data)[3].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float mult = gamma[feature_map_id] * invvar[feature_map_id];
				float add = beta[feature_map_id] - mult * mean[feature_map_id];

				const float * current_in_it = in_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
				const float * current_in_it_end = current_in_it + neuron_count_per_feature_map;

				float * current_out_it = out_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);

				for(; current_in_it != current_in_it_end; ++current_in_it, ++current_out_it)
				{
					float input_val = *current_in_it;
					float output_val = input_val * mult + add;
					*current_out_it = output_val;
				}
			}
		}

		int batch_norm_layer_tester_plain::get_input_index_layer_can_write(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}
	}
}
