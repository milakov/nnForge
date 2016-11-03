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

#include "hyperbolic_tangent_layer_updater_plain.h"

#include "../hyperbolic_tangent_layer.h"

namespace nnforge
{
	namespace plain
	{
		std::string hyperbolic_tangent_layer_updater_plain::get_type_name() const
		{
			return hyperbolic_tangent_layer::layer_type_name;
		}

		void hyperbolic_tangent_layer_updater_plain::run_forward_propagation(
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
			const int elem_count = static_cast<int>(entry_count * output_configuration_specific.get_neuron_count());
			float * const out_it = *output_buffer;
			const float * const in_it = *input_buffers[0];

			std::shared_ptr<const hyperbolic_tangent_layer> layer_derived = std::dynamic_pointer_cast<const hyperbolic_tangent_layer>(layer_schema);
			const float hyperbolic_tangent_steepness2 = layer_derived->steepness * 2.0F;
			const float hyperbolic_tangent_major_multiplier = layer_derived->scale;

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
			{
				float inp = *(in_it + i);
				float inp2 = expf(inp * hyperbolic_tangent_steepness2);
				float res = (inp2 - 1.0F) / (inp2 + 1.0F) * hyperbolic_tangent_major_multiplier;
				*(out_it + i) = res;
			}
		}

		void hyperbolic_tangent_layer_updater_plain::run_backward_data_propagation(
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
			const int elem_count = static_cast<int>(entry_count * output_configuration_specific.get_neuron_count());
			float * const in_err_it = *input_errors_buffer;
			const float * const out_it = *output_neurons_buffer;
			const float * const out_err_it = *output_errors_buffer;

			std::shared_ptr<const hyperbolic_tangent_layer> layer_derived = std::dynamic_pointer_cast<const hyperbolic_tangent_layer>(layer_schema);
			const float hyperbolic_tangent_major_multiplier_reverse = 1.0F / layer_derived->scale;
			const float hyperbolic_tangent_steepness3 = layer_derived->steepness * layer_derived->scale;
			if (add_update_to_destination)
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < elem_count; ++i)
				{
					float out_neuron = *(out_it + i);
					float normalized_value = out_neuron * hyperbolic_tangent_major_multiplier_reverse;
					float der1st = hyperbolic_tangent_steepness3 * (1.0F - (normalized_value * normalized_value));
					*(in_err_it + i) += *(out_err_it + i) * der1st;
				}
			}
			else
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < elem_count; ++i)
				{
					float out_neuron = *(out_it + i);
					float normalized_value = out_neuron * hyperbolic_tangent_major_multiplier_reverse;
					float der1st = hyperbolic_tangent_steepness3 * (1.0F - (normalized_value * normalized_value));
					*(in_err_it + i) = *(out_err_it + i) * der1st;
				}
			}
		}

		int hyperbolic_tangent_layer_updater_plain::get_input_index_layer_can_write(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}

		bool hyperbolic_tangent_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool hyperbolic_tangent_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return true;
		}
	}
}
