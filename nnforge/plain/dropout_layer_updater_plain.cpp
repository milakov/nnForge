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

#include "dropout_layer_updater_plain.h"

#include "../dropout_layer.h"
#include "../nn_types.h"

#include <cstring>

namespace nnforge
{
	namespace plain
	{
		dropout_layer_updater_plain::dropout_layer_updater_plain()
			: gen(rnd::get_random_generator())
		{
		}

		dropout_layer_updater_plain::~dropout_layer_updater_plain()
		{
		}

		std::string dropout_layer_updater_plain::get_type_name() const
		{
			return dropout_layer::layer_type_name;
		}

		void dropout_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			unsigned char * keep_elem_ptr = *temporary_per_entry_buffer;

			nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
			const float dropout_rate = layer_derived->dropout_rate;
			const float keep_rate = 1.0F - dropout_rate;
			const float mult = 1.0F / keep_rate;

			const int total_workload = output_configuration_specific.get_neuron_count() * entry_count;

			nnforge_uniform_real_distribution<float> dist(0.0F, 1.0F);

			for(int i = 0; i < total_workload; ++i)
				keep_elem_ptr[i] = (dist(gen) <= keep_rate ? (unsigned char)1 : (unsigned char)0);

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int elem_id = workload_id;
					*(out_it_global + elem_id) = *(in_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
				}
			}
		}

		void dropout_layer_updater_plain::run_backward_data_propagation(
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
			float * const in_err_it_global = *input_errors_buffer;
			const float * const out_err_it_global = *output_errors_buffer;
			const unsigned char * keep_elem_ptr = *temporary_per_entry_buffer;

			nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
			const float dropout_rate = layer_derived->dropout_rate;
			const float keep_rate = 1.0F - dropout_rate;
			const float mult = 1.0F / keep_rate;

			const int total_workload = output_configuration_specific.get_neuron_count() * entry_count;

			if (add_update_to_destination)
			{
				#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr)
				{
					#pragma omp for schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int elem_id = workload_id;
						*(in_err_it_global + elem_id) += *(out_err_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
					}
				}
			}
			else
			{
				#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr)
				{
					#pragma omp for schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int elem_id = workload_id;
						*(in_err_it_global + elem_id) = *(out_err_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
					}
				}
			}
		}

		bool dropout_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool dropout_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		size_t dropout_layer_updater_plain::get_temporary_per_entry_buffer_size(
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return output_configuration_specific.get_neuron_count() * sizeof(unsigned char);
		}
	}
}
