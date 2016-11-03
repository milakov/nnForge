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

#include "layer_updater_plain.h"
#include "../rnd.h"

namespace nnforge
{
	namespace plain
	{
		class dropout_layer_updater_plain : public layer_updater_plain
		{
		public:
			dropout_layer_updater_plain();

			virtual ~dropout_layer_updater_plain() = default;

			virtual std::string get_type_name() const;

			virtual void run_forward_propagation(
				plain_buffer::ptr output_buffer,
				const std::vector<plain_buffer::const_ptr>& input_buffers,
				plain_buffer::ptr temporary_working_fixed_buffer,
				plain_buffer::ptr temporary_working_per_entry_buffer,
				plain_buffer::ptr temporary_per_entry_buffer,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				layer_data::const_ptr data,
				layer_data_custom::const_ptr data_custom,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				const std::set<layer_action>& actions,
				unsigned int entry_count) const;

			virtual void run_backward_data_propagation(
				unsigned int input_index,
				plain_buffer::ptr input_errors_buffer,
				plain_buffer::const_ptr output_errors_buffer,
				const std::vector<plain_buffer::const_ptr>& input_neurons_buffers,
				plain_buffer::const_ptr output_neurons_buffer,
				plain_buffer::ptr temporary_working_fixed_buffer,
				plain_buffer::ptr temporary_working_per_entry_buffer,
				plain_buffer::ptr temporary_per_entry_buffer,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				layer_data::const_ptr data,
				layer_data_custom::const_ptr data_custom,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				const bool add_update_to_destination,
				const std::set<layer_action>& actions,
				unsigned int entry_count) const;

			virtual int get_input_index_layer_can_write(
				const layer_action& action,
				const std::set<layer_action>& actions,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual bool is_backward_data_dependent_on_input_buffer(
				unsigned int action_input_index,
				unsigned int data_input_index,
				const std::set<layer_action>& actions,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual bool is_backward_data_dependent_on_output_buffer(
				unsigned int action_input_index,
				const std::set<layer_action>& actions,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual size_t get_temporary_per_entry_buffer_size(
				const std::set<layer_action>& actions,
				plain_running_configuration::const_ptr plain_config,
				layer::const_ptr layer_schema,
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;

		private:
			mutable random_generator gen;
		};
	}
}
