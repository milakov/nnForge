/*
 *  Copyright 2011-2014 Maxim Milakov
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

		void layer_updater_plain::update_buffer_configuration(
			buffer_plain_size_configuration& buffer_configuration,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			std::vector<std::pair<unsigned int, bool> > buffer_sizes_per_entry_aligned = get_elem_count_and_per_entry_flag_additional_buffers(
				layer_schema,
				input_configuration_specific,
				output_configuration_specific,
				plain_config,
				backprop_required);
			for(std::vector<std::pair<unsigned int, bool> >::const_iterator it = buffer_sizes_per_entry_aligned.begin(); it != buffer_sizes_per_entry_aligned.end(); ++it)
			{
				size_t s = static_cast<size_t>(it->first) * sizeof(float);
				if (it->second)
					buffer_configuration.add_per_entry_buffer(s);
				else
					buffer_configuration.add_constant_buffer(s);
			}

			buffer_configuration.add_per_entry_buffer(output_configuration_specific.get_neuron_count() * sizeof(float));

			if (backprop_required && !is_in_place_backprop())
				buffer_configuration.add_per_entry_buffer(input_configuration_specific.get_neuron_count() * sizeof(float));
		}

		void layer_updater_plain::update_buffer_configuration(
			buffer_plain_size_configuration& buffer_configuration,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required,
			unsigned int updater_entry_count) const
		{
			std::vector<std::pair<unsigned int, bool> > buffer_sizes_per_entry_aligned = get_elem_count_and_per_entry_flag_additional_buffers(
				layer_schema,
				input_configuration_specific,
				output_configuration_specific,
				plain_config,
				backprop_required);
			for(std::vector<std::pair<unsigned int, bool> >::const_iterator it = buffer_sizes_per_entry_aligned.begin(); it != buffer_sizes_per_entry_aligned.end(); ++it)
			{
				size_t s = static_cast<size_t>(it->first) * sizeof(float);
				if (it->second)
					buffer_configuration.add_constant_buffer(s * updater_entry_count);
				else
					buffer_configuration.add_constant_buffer(s * updater_entry_count);
			}

			buffer_configuration.add_constant_buffer(output_configuration_specific.get_neuron_count() * sizeof(float) * updater_entry_count);

			if (backprop_required && !is_in_place_backprop())
				buffer_configuration.add_constant_buffer(input_configuration_specific.get_neuron_count() * sizeof(float) * updater_entry_count);
		}

		std::vector<std::pair<unsigned int, bool> > layer_updater_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			return std::vector<std::pair<unsigned int, bool> >();
		}

		updater_additional_buffer_set layer_updater_plain::allocate_additional_buffers(
			unsigned int updater_entry_count,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			updater_additional_buffer_set res;

			std::vector<std::pair<unsigned int, bool> > buffer_sizes_per_entry_aligned = get_elem_count_and_per_entry_flag_additional_buffers(
				layer_schema,
				input_configuration_specific,
				output_configuration_specific,
				plain_config,
				backprop_required);

			for(std::vector<std::pair<unsigned int, bool> >::const_iterator it = buffer_sizes_per_entry_aligned.begin(); it != buffer_sizes_per_entry_aligned.end(); ++it)
				res.additional_buffers.push_back(additional_buffer_smart_ptr(new std::vector<float>(it->first * (it->second ? updater_entry_count : 1))));

			res.output_neurons_buffer = additional_buffer_smart_ptr(new std::vector<float>(output_configuration_specific.get_neuron_count() * updater_entry_count));

			if (backprop_required && !is_in_place_backprop())
				res.input_errors_buffer = additional_buffer_smart_ptr(new std::vector<float>(input_configuration_specific.get_neuron_count() * updater_entry_count));

			return res;
		}

		void layer_updater_plain::update_weights(
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			layer_data_smart_ptr gradient,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id) const
		{
		}
	}
}
