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

#include "layer_tester_plain.h"

namespace nnforge
{
	namespace plain
	{
		layer_tester_plain::layer_tester_plain()
		{
		}

		layer_tester_plain::~layer_tester_plain()
		{
		}

		void layer_tester_plain::update_buffer_configuration(
			buffer_plain_size_configuration& buffer_configuration,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			unsigned int tiling_factor) const
		{
			std::vector<std::pair<unsigned int, bool> > buffer_sizes_per_entry_aligned = get_elem_count_and_per_entry_flag_additional_buffers(
				layer_schema,
				input_configuration_specific,
				output_configuration_specific,
				plain_config);
			for(std::vector<std::pair<unsigned int, bool> >::const_iterator it = buffer_sizes_per_entry_aligned.begin(); it != buffer_sizes_per_entry_aligned.end(); ++it)
			{
				size_t s = static_cast<size_t>(it->first) * sizeof(float);
				if (it->second)
					buffer_configuration.add_per_entry_buffer(s * tiling_factor);
				else
					buffer_configuration.add_constant_buffer(s);
			}
		}

		std::vector<std::pair<unsigned int, bool> > layer_tester_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config) const
		{
			return std::vector<std::pair<unsigned int, bool> >();
		}

		additional_buffer_set layer_tester_plain::allocate_additional_buffers(
			unsigned int max_entry_count,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config) const
		{
			additional_buffer_set res;

			std::vector<std::pair<unsigned int, bool> > buffer_sizes_per_entry_aligned = get_elem_count_and_per_entry_flag_additional_buffers(
				layer_schema,
				input_configuration_specific,
				output_configuration_specific,
				plain_config);

			for(std::vector<std::pair<unsigned int, bool> >::const_iterator it = buffer_sizes_per_entry_aligned.begin(); it != buffer_sizes_per_entry_aligned.end(); ++it)
				res.push_back(additional_buffer_smart_ptr(new std::vector<float>(it->first * (it->second ? max_entry_count : 1))));

			return res;
		}

		additional_buffer_smart_ptr layer_tester_plain::get_output_buffer(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers) const
		{
			return input_buffer;
		}
	}
}
