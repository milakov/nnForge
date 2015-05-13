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

#include "sigmoid_layer_tester_plain.h"

#include "../sigmoid_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace plain
	{
		sigmoid_layer_tester_plain::sigmoid_layer_tester_plain()
		{
		}

		sigmoid_layer_tester_plain::~sigmoid_layer_tester_plain()
		{
		}

		const boost::uuids::uuid& sigmoid_layer_tester_plain::get_uuid() const
		{
			return sigmoid_layer::layer_guid;
		}

		void sigmoid_layer_tester_plain::test(
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
			nnforge_shared_ptr<const sigmoid_layer> layer_derived = nnforge_dynamic_pointer_cast<const sigmoid_layer>(layer_schema);
			const std::vector<float>::iterator in_it = input_buffer->begin();
			
			if (layer_derived->affected_feature_map_id_list.empty())
			{
				const int elem_count = static_cast<int>(entry_count * input_configuration_specific.get_neuron_count());

				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < elem_count; ++i)
				{
					float inp = *(in_it + i);
					float res = 1.0F / (expf(-inp) + 1.0F);
					*(in_it + i) = res;
				}
			}
			else
			{
				const int affected_feature_map_count = static_cast<int>(layer_derived->affected_feature_map_id_list.size());
				const int elem_count = static_cast<int>(entry_count * affected_feature_map_count);
				const int neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
				const int feature_map_count = input_configuration_specific.feature_map_count;
				const std::vector<unsigned int>::const_iterator affected_feature_map_id_it = layer_derived->affected_feature_map_id_list.begin();

				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int elem_id = 0; elem_id < elem_count; ++elem_id)
				{
					unsigned int entry_id = elem_id / affected_feature_map_count;
					unsigned int feature_map_config_id = elem_id - affected_feature_map_count * entry_id;
					unsigned int feature_map_id = *(affected_feature_map_id_it + feature_map_config_id);

					const std::vector<float>::iterator in_it2 = in_it + ((entry_id * feature_map_count + feature_map_id) * neuron_count_per_feature_map);

					for(int i = 0; i < neuron_count_per_feature_map; ++i)
					{
						float inp = *(in_it2 + i);
						float res = 1.0F / (expf(-inp) + 1.0F);
						*(in_it2 + i) = res;
					}
				}
			}
		}
	}
}
