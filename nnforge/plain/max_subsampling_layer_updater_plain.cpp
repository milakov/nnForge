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

#include "max_subsampling_layer_updater_plain.h"

#include "../max_subsampling_layer.h"
#include "../nn_types.h"
#include "../neural_network_exception.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int max_subsampling_layer_updater_plain::max_dimension_count = 4;

		max_subsampling_layer_updater_plain::max_subsampling_layer_updater_plain()
		{
		}

		max_subsampling_layer_updater_plain::~max_subsampling_layer_updater_plain()
		{
		}

		std::string max_subsampling_layer_updater_plain::get_type_name() const
		{
			return max_subsampling_layer::layer_type_name;
		}

		void max_subsampling_layer_updater_plain::run_forward_propagation(
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
			std::vector<unsigned int> input_dimension_sizes = input_configuration_specific_list[0].dimension_sizes;
			if (input_dimension_sizes.empty())
				input_dimension_sizes.push_back(1);
			std::vector<unsigned int> output_dimension_sizes = output_configuration_specific.dimension_sizes;
			if (output_dimension_sizes.empty())
				output_dimension_sizes.push_back(1);

			nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

			if (layer_derived->tiling)
				throw neural_network_exception("max_subsampling_layer_updater_plain is not able to run for max subsampling layer with tiling");

			for(std::vector<bool>::const_iterator it = layer_derived->round_ups.begin(); it != layer_derived->round_ups.end(); ++it)
				if (*it)
					throw neural_network_exception("round up is not implemented for max_subsampling_layer_tester_plain");

			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			unsigned int * const max_indexes_it_global = *temporary_per_entry_buffer;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			std::vector<unsigned int> strides = layer_derived->strides;
			if (strides.empty())
				strides.push_back(1);
			std::vector<unsigned int> subsampling_sizes = layer_derived->subsampling_sizes;
			if (subsampling_sizes.empty())
				subsampling_sizes.push_back(1);
			const unsigned int feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
			subsampling_sizes.push_back(feature_map_subsampling_size);
			const unsigned int entry_subsampling_size = layer_derived->entry_subsampling_size;
			subsampling_sizes.push_back(entry_subsampling_size);
			const unsigned int subsampling_dimension_count = static_cast<unsigned int>(subsampling_sizes.size());
			const unsigned int spatial_dimension_count = static_cast<unsigned int>(output_dimension_sizes.size());
			std::vector<unsigned int> input_slices(subsampling_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < subsampling_dimension_count - 1; ++i)
			{
				int dimension_size = (i < spatial_dimension_count) ? input_dimension_sizes[i] : input_configuration_specific_list[0].feature_map_count;
				input_slices[i + 1] = input_slices[i] * dimension_size;
			}
			unsigned int subsampling_elem_count = 1;
			for(unsigned int i = 0; i < subsampling_dimension_count; ++i)
				subsampling_elem_count *= subsampling_sizes[i];
			const unsigned int const_subsampling_elem_count = subsampling_elem_count;
			const unsigned int output_feature_map_count = output_configuration_specific.feature_map_count;
			const bool is_min = layer_derived->is_min;

			std::vector<unsigned int> current_local_input_position(subsampling_dimension_count, 0);
			std::vector<unsigned int> offset_list(subsampling_elem_count);
			for(unsigned int i = 1; i < subsampling_elem_count; ++i)
			{
				int offset = 0;
				for(unsigned int j = 0; j < subsampling_dimension_count; ++j)
				{
					offset += static_cast<int>(input_slices[j]);
					if ((++current_local_input_position[j]) < subsampling_sizes[j])
					{
						offset_list[i] = offset_list[i-1] + offset;
						break;
					}
					current_local_input_position[j] = 0;
					offset -= static_cast<int>(subsampling_sizes[j] * input_slices[j]);
				}
			}

			const int total_workload = entry_count * output_configuration_specific.feature_map_count;
			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator strides_it = strides.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int output_entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (output_entry_id * output_feature_map_count);

					const int in_base_offset = (output_entry_id * entry_subsampling_size * input_neuron_count) + (output_feature_map_id * feature_map_subsampling_size * input_neuron_count_per_feature_map);
					float * out_it_base = out_it_global + (output_entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					unsigned int * max_indexes_it_base = max_indexes_it_global + (output_entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

					std::fill_n(current_output_position.begin(), spatial_dimension_count, 0);
					unsigned int * max_indexes_it = max_indexes_it_base;
					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++max_indexes_it)
					{
						// Define the starting position of the first input elem
						int in_offset = in_base_offset;
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
							in_offset += current_output_position[i] * (*(strides_it + i)) * (*(input_slices_it + i));

						unsigned int max_index = 0;
						float best_val = is_min ? 1.0e37F : -1.0e37F;
						for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
						{
							int current_offset = in_offset + *(offset_list_it + i);
							float new_val = *(in_it_global + current_offset);
							if ((i == 0) || (((new_val > best_val) && !is_min) || ((new_val < best_val) && is_min)))
							{
								best_val = new_val;
								max_index = current_offset;
							}
						}
						*out_it = best_val;
						*max_indexes_it = max_index;

						// Go to the next output element
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
						{
							if ((++current_output_position[i]) < *( dimension_sizes_it + i))
								break;
							current_output_position[i] = 0;
						}
					}
				}
			}
		}

		void max_subsampling_layer_updater_plain::run_backward_data_propagation(
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
			const unsigned int * const max_indexes_it_global = *temporary_per_entry_buffer;

			nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);
			const unsigned int entry_subsampling_size = layer_derived->entry_subsampling_size;

			if (!add_update_to_destination)
			{
				const int total_clean_workload = entry_count * entry_subsampling_size * input_configuration_specific_list[0].get_neuron_count();
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int workload_id = 0; workload_id < total_clean_workload; ++workload_id)
				{
					*(in_err_it_global + workload_id) = 0.0F;
				}
			}

			const int total_workload = entry_count * output_configuration_specific.get_neuron_count();

			if (add_update_to_destination)
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					unsigned int max_index = *(max_indexes_it_global + workload_id);
					float err = *(out_err_it_global + workload_id);
					*(in_err_it_global + max_index) += err;
				}
			}
			else
			{
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					unsigned int max_index = *(max_indexes_it_global + workload_id);
					float err = *(out_err_it_global + workload_id);
					*(in_err_it_global + max_index) = err;
				}
			}
		}

		bool max_subsampling_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool max_subsampling_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		size_t max_subsampling_layer_updater_plain::get_temporary_per_entry_buffer_size(
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return output_configuration_specific.get_neuron_count() * sizeof(unsigned int);
		}
	}
}
