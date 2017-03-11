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

#include "dropout_layer_updater_plain.h"

#include "../dropout_layer.h"

#include <cstring>

namespace nnforge
{
	namespace plain
	{
		dropout_layer_updater_plain::dropout_layer_updater_plain()
			: gen(rnd::get_random_generator())
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

			std::shared_ptr<const dropout_layer> layer_derived = std::dynamic_pointer_cast<const dropout_layer>(layer_schema);
			const float dropout_rate = layer_derived->dropout_rate;
			const float keep_rate = 1.0F - dropout_rate;
			const float mult = 1.0F / keep_rate;

			std::uniform_real_distribution<float> dist(0.0F, 1.0F);

			const int total_workload = (layer_derived->per_feature_map ? output_configuration_specific.feature_map_count : output_configuration_specific.get_neuron_count()) * entry_count;
			for(int i = 0; i < total_workload; ++i)
				keep_elem_ptr[i] = (dist(gen) <= keep_rate ? (unsigned char)1 : (unsigned char)0);

			if (layer_derived->per_feature_map)
			{
				const int elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
				#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int base_elem_id = workload_id * elem_count_per_feature_map;
					float * out_it = out_it_global + base_elem_id;
					if (keep_elem_ptr[workload_id])
					{
						const float * in_it = in_it_global + base_elem_id;
						for(int elem_id = 0; elem_id < elem_count_per_feature_map; ++elem_id)
							*(out_it + elem_id) = *(in_it + elem_id) * mult;
					}
					else
					{
						std::fill_n(out_it, elem_count_per_feature_map, 0.0F);
					}
				}
			}
			else
			{
				#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
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

			std::shared_ptr<const dropout_layer> layer_derived = std::dynamic_pointer_cast<const dropout_layer>(layer_schema);
			const float dropout_rate = layer_derived->dropout_rate;
			const float keep_rate = 1.0F - dropout_rate;
			const float mult = 1.0F / keep_rate;

			const int total_workload = (layer_derived->per_feature_map ? output_configuration_specific.feature_map_count : output_configuration_specific.get_neuron_count()) * entry_count;

			if (add_update_to_destination)
			{
				if (layer_derived->per_feature_map)
				{
					const int elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
					#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int base_elem_id = workload_id * elem_count_per_feature_map;
						float * in_err_it = in_err_it_global + base_elem_id;
						if (keep_elem_ptr[workload_id])
						{
							const float * out_err_it = out_err_it_global + base_elem_id;
							for(int elem_id = 0; elem_id < elem_count_per_feature_map; ++elem_id)
								*(in_err_it + elem_id) += *(out_err_it + elem_id) * mult;
						}
					}
				}
				else
				{
					#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int elem_id = workload_id;
						*(in_err_it_global + elem_id) += *(out_err_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
					}
				}
			}
			else
			{
				if (layer_derived->per_feature_map)
				{
					const int elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
					#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int base_elem_id = workload_id * elem_count_per_feature_map;
						float * in_err_it = in_err_it_global + base_elem_id;
						if (keep_elem_ptr[workload_id])
						{
							const float * out_err_it = out_err_it_global + base_elem_id;
							for(int elem_id = 0; elem_id < elem_count_per_feature_map; ++elem_id)
								*(in_err_it + elem_id) = *(out_err_it + elem_id) * mult;
						}
						else
						{
							std::fill_n(in_err_it, elem_count_per_feature_map, 0.0F);
						}
					}
				}
				else
				{
					#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr) schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int elem_id = workload_id;
						*(in_err_it_global + elem_id) = *(out_err_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
					}
				}
			}
		}

		int dropout_layer_updater_plain::get_input_index_layer_can_write(
			const layer_action& action,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
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
			std::shared_ptr<const dropout_layer> layer_derived = std::dynamic_pointer_cast<const dropout_layer>(layer_schema);
			return (layer_derived->per_feature_map ? output_configuration_specific.feature_map_count : output_configuration_specific.get_neuron_count()) * sizeof(unsigned char);
		}
	}
}
