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

#pragma once

#include "layer_tester_plain.h"

namespace nnforge
{
	namespace plain
	{
		class add_layer_tester_plain : public layer_tester_plain
		{
		public:
			add_layer_tester_plain();

			virtual ~add_layer_tester_plain();

			virtual std::string get_type_name() const;

			virtual void run_forward_propagation(
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
				unsigned int entry_count) const;

			virtual int get_input_index_layer_can_write(
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;
		};
	}
}
