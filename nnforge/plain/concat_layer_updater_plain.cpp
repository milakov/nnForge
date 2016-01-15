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

#include "concat_layer_updater_plain.h"

#include "../concat_layer.h"
#include <cstring>

namespace nnforge
{
	namespace plain
	{
		concat_layer_updater_plain::concat_layer_updater_plain()
		{
		}

		concat_layer_updater_plain::~concat_layer_updater_plain()
		{
		}

		std::string concat_layer_updater_plain::get_type_name() const
		{
			return concat_layer::layer_type_name;
		}

		void concat_layer_updater_plain::run_forward_propagation(
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

		void concat_layer_updater_plain::run_backward_data_propagation(
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
			unsigned int entry_count) const
		{
			unsigned int offset = 0;
			for(unsigned int i = 0; i < input_index; ++i)
				offset += input_configuration_specific_list[i].get_neuron_count();

			for(unsigned int entry_id = 0; entry_id < entry_count; ++entry_id)
			{
				unsigned int input_neuron_count = input_configuration_specific_list[input_index].get_neuron_count();
				const float * out_err = (const float *)(*output_errors_buffer) + entry_id * output_configuration_specific.get_neuron_count() + offset;
				float * in_err = (float *)(*input_errors_buffer) + entry_id * input_neuron_count;
				if (add_update_to_destination)
				{
					for(unsigned int i = 0; i < input_neuron_count; ++i)
						in_err[i] += out_err[i];
				}
				else
				{
					memcpy(in_err, out_err, input_neuron_count * sizeof(float));
				}
			}
		}

		bool concat_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
			unsigned int action_input_index,
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		bool concat_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}
	}
}
