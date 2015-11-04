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

#include "layer_updater_plain.h"

#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		layer_updater_plain::layer_updater_plain()
		{
		}

		layer_updater_plain::~layer_updater_plain()
		{
		}

		void layer_updater_plain::run_backward_data_propagation(
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
			throw neural_network_exception((boost::format("run_backward_data_propagation is not implemented for layer %1%") % layer_schema->instance_name).str());
		}

		void layer_updater_plain::run_backward_weights_propagation(
			const std::vector<plain_buffer::const_ptr>& input_neurons_buffers,
			plain_buffer::const_ptr output_errors_buffer,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_buffer::ptr temporary_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::ptr gradient,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			const std::set<layer_action>& actions,
			unsigned int entry_count) const
		{
			throw neural_network_exception((boost::format("run_backward_data_propagation is not implemented for layer %1%") % layer_schema->instance_name).str());
		}

		int layer_updater_plain::get_input_index_layer_can_write(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_input_index_layer_can_write called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return -1;
		}

		size_t layer_updater_plain::get_temporary_working_fixed_buffer_size(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_working_fixed_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return 0;
		}

		size_t layer_updater_plain::get_temporary_working_per_entry_buffer_size(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_working_per_entry_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return 0;
		}

		size_t layer_updater_plain::get_temporary_per_entry_buffer_size(
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::forward)) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_per_entry_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % layer_action(layer_action::forward).str()).str());

			return 0;
		}

		bool layer_updater_plain::is_backward_data_dependent_on_input_buffer(
			unsigned int action_input_index,
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_input_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return true;
		}

		bool layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_output_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return true;
		}

		bool layer_updater_plain::is_backward_data_dependent_on_temporary_per_entry_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_temporary_per_entry_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return (get_temporary_per_entry_buffer_size(actions, plain_config, layer_schema, input_configuration_specific_list, output_configuration_specific) != 0);
		}

		bool layer_updater_plain::is_backward_weights_dependent_on_input_buffer(
			unsigned int data_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::backward_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_weights_dependent_on_input_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_weights).str()).str());

			return true;
		}

		bool layer_updater_plain::is_backward_weights_dependent_on_temporary_per_entry_buffer(
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			if (actions.find(layer_action(layer_action::backward_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_weights_dependent_on_temporary_per_entry_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_weights).str()).str());

			return (get_temporary_per_entry_buffer_size(actions, plain_config, layer_schema, input_configuration_specific_list, output_configuration_specific) != 0);
		}
	}
}
