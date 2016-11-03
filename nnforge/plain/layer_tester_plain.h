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

#pragma once

#include <memory>
#include <boost/uuid/uuid.hpp>

#include "../layer.h"

#include "plain_running_configuration.h"
#include "buffer_plain_size_configuration.h"
#include "plain_buffer.h"

namespace nnforge
{
	namespace plain
	{
		class layer_tester_plain
		{
		public:
			typedef std::shared_ptr<layer_tester_plain> ptr;
			typedef std::shared_ptr<const layer_tester_plain> const_ptr;

			virtual ~layer_tester_plain() = default;

			virtual std::string get_type_name() const = 0;

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
				unsigned int entry_count) const = 0;

			virtual int get_input_index_layer_can_write(
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual size_t get_temporary_working_fixed_buffer_size(
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual size_t get_temporary_working_per_entry_buffer_size(
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

		protected:
			layer_tester_plain() = default;

		private:
			layer_tester_plain(const layer_tester_plain&) = delete;
			layer_tester_plain& operator =(const layer_tester_plain&) = delete;
		};
	}
}
