/*
 *  Copyright 2011-2013 Maxim Milakov
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
		maxout_layer_tester_plain::maxout_layer_tester_plain()
		{
		}

		maxout_layer_tester_plain::~maxout_layer_tester_plain()
		{
		}

		const boost::uuids::uuid& maxout_layer_tester_plain::get_uuid() const
		{
			return maxout_layer::layer_guid;
		}

		void maxout_layer_tester_plain::test(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const std::vector<float>::const_iterator in_it_global = input_buffer->begin();
			const std::vector<float>::iterator out_it_global = additional_buffers[0]->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			std::tr1::shared_ptr<const maxout_layer> layer_derived = std::tr1::dynamic_pointer_cast<const maxout_layer>(layer_schema);
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

					std::vector<float>::const_iterator in_it_base = in_it_global + (entry_id * input_neuron_count) + (output_feature_map_id * input_neuron_count_per_feature_map);
					std::vector<float>::iterator out_it_base = out_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

					for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++in_it_base)
					{
						std::vector<float>::const_iterator in_it = in_it_base;
						float current_max = *in_it;
						for(int i = 1; i < feature_map_subsampling_size; ++i)
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

		additional_buffer_smart_ptr maxout_layer_tester_plain::get_output_buffer(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers) const
		{
			return additional_buffers[0];
		}

		std::vector<std::pair<unsigned int, bool> > maxout_layer_tester_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			res.push_back(std::make_pair<unsigned int, bool>(output_configuration_specific.get_neuron_count(), true));

			return res;
		}
	}
}
