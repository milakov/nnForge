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

#include "concat_layer_tester_plain.h"

#include "../concat_layer.h"

#include <cstring>

namespace nnforge
{
	namespace plain
	{
		std::string concat_layer_tester_plain::get_type_name() const
		{
			return concat_layer::layer_type_name;
		}

		void concat_layer_tester_plain::run_forward_propagation(
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
			for(unsigned int entry_id = 0; entry_id < entry_count; ++entry_id)
			{
				float *dst = (float *)*output_buffer + entry_id * output_configuration_specific.get_neuron_count();
				for(unsigned int i = 0; i < static_cast<unsigned int>(input_configuration_specific_list.size()); ++i)
				{
					unsigned int input_neuron_count = input_configuration_specific_list[i].get_neuron_count();
					memcpy(
						dst,
						(const float *)(*input_buffers[i]) + entry_id * input_neuron_count,
						input_neuron_count * sizeof(float));
					dst += input_neuron_count;
				}
			}
		}
	}
}
