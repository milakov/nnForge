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

#include "average_subsampling_layer_updater_plain.h"

#include "../average_subsampling_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int average_subsampling_layer_updater_plain::max_dimension_count = 4;

		average_subsampling_layer_updater_plain::average_subsampling_layer_updater_plain()
		{
		}

		average_subsampling_layer_updater_plain::~average_subsampling_layer_updater_plain()
		{
		}

		std::string average_subsampling_layer_updater_plain::get_type_name() const
		{
			return average_subsampling_layer::layer_type_name;
		}

		void average_subsampling_layer_updater_plain::run_forward_propagation(
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
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);
			std::vector<unsigned int> subsampling_sizes;
			for(unsigned int i = 0; i < static_cast<unsigned int>(layer_derived->subsampling_sizes.size()); ++i)
				subsampling_sizes.push_back(layer_derived->get_subsampling_size(i, input_dimension_sizes[i], output_dimension_sizes[i]));
			if (subsampling_sizes.empty())
				subsampling_sizes.push_back(1);
			const unsigned int feature_map_subsampling_size = layer_derived->get_fm_subsampling_size(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
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
			const float mult = layer_derived->get_effective_alpha(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
			const unsigned int output_feature_map_count = output_configuration_specific.feature_map_count;

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
			const std::vector<unsigned int>::const_iterator subsampling_sizes_it = subsampling_sizes.begin();
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

					const float * in_it_base = in_it_global + (output_entry_id * entry_subsampling_size * input_neuron_count) + (output_feature_map_id * feature_map_subsampling_size * input_neuron_count_per_feature_map);
					float * out_it_base = out_it_global + (output_entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

					std::fill_n(current_output_position.begin(), spatial_dimension_count, 0);
					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						// Define the starting position of the first input elem
						const float * in_it = in_it_base;
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
							in_it += current_output_position[i] * (*(subsampling_sizes_it + i)) * (*(input_slices_it + i));

						float sum = 0.0F;
						for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
						{
							sum += *(in_it + (*(offset_list_it + i)));
						}
						*out_it = sum * mult;

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

		void average_subsampling_layer_updater_plain::run_backward_data_propagation(
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
			std::vector<unsigned int> input_dimension_sizes = input_configuration_specific_list[0].dimension_sizes;
			if (input_dimension_sizes.empty())
				input_dimension_sizes.push_back(1);
			std::vector<unsigned int> output_dimension_sizes = output_configuration_specific.dimension_sizes;
			if (output_dimension_sizes.empty())
				output_dimension_sizes.push_back(1);
			float * const in_err_it_global = *input_errors_buffer;
			const float * const out_err_it_global = *output_errors_buffer;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);
			std::vector<unsigned int> subsampling_sizes;
			for(unsigned int i = 0; i < static_cast<unsigned int>(layer_derived->subsampling_sizes.size()); ++i)
				subsampling_sizes.push_back(layer_derived->get_subsampling_size(i, input_dimension_sizes[i], output_dimension_sizes[i]));
			if (subsampling_sizes.empty())
				subsampling_sizes.push_back(1);
			const unsigned int feature_map_subsampling_size = layer_derived->get_fm_subsampling_size(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
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
			const float mult = layer_derived->get_effective_alpha(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
			const unsigned int output_feature_map_count = output_configuration_specific.feature_map_count;

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

			if (!add_update_to_destination)
			{
				bool exact_subsampling = (input_configuration_specific_list[0].feature_map_count == feature_map_subsampling_size * output_configuration_specific.feature_map_count);
				for(unsigned int i = 0; i < static_cast<unsigned int>(subsampling_sizes.size()); ++i)
				{
					if (input_dimension_sizes[i] != subsampling_sizes[i] * output_dimension_sizes[i])
					{
						exact_subsampling = false;
						break;
					}
				}
				if (!exact_subsampling)
				{
					const int total_clean_workload = entry_count * entry_subsampling_size * input_neuron_count;
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int workload_id = 0; workload_id < total_clean_workload; ++workload_id)
					{
						*(in_err_it_global + workload_id) = 0.0F;
					}
				}
			}

			const int total_workload = entry_count * output_configuration_specific.feature_map_count;
			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator subsampling_sizes_it = subsampling_sizes.begin();
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

					float * in_err_it_base = in_err_it_global + (output_entry_id * entry_subsampling_size * input_neuron_count) + (output_feature_map_id * feature_map_subsampling_size * input_neuron_count_per_feature_map);
					const float * out_err_it_base = out_err_it_global + (output_entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

					std::fill_n(current_output_position.begin(), spatial_dimension_count, 0);
					for(const float * out_it = out_err_it_base; out_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						// Define the starting position of the first input elem
						float * in_it = in_err_it_base;
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
							in_it += current_output_position[i] * (*(subsampling_sizes_it + i)) * (*(input_slices_it + i));

						float err = *out_it * mult;
						if (add_update_to_destination)
						{
							for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
								*(in_it + (*(offset_list_it + i))) += err;
						}
						else
						{
							for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
								*(in_it + (*(offset_list_it + i))) = err;
						}

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

		bool average_subsampling_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool average_subsampling_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
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
