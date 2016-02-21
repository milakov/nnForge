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

#include "upsampling_layer_tester_plain.h"

#include "../upsampling_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int upsampling_layer_tester_plain::max_dimension_count = 4;

		upsampling_layer_tester_plain::upsampling_layer_tester_plain()
		{
		}

		upsampling_layer_tester_plain::~upsampling_layer_tester_plain()
		{
		}

		std::string upsampling_layer_tester_plain::get_type_name() const
		{
			return upsampling_layer::layer_type_name;
		}

		void upsampling_layer_tester_plain::run_forward_propagation(
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
			nnforge_shared_ptr<const upsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const upsampling_layer>(layer_schema);
			std::vector<unsigned int> upsampling_sizes = layer_derived->upsampling_sizes;
			if (upsampling_sizes.empty())
				upsampling_sizes.push_back(1);
			const unsigned int feature_map_upsampling_size = layer_derived->feature_map_upsampling_size;
			upsampling_sizes.push_back(feature_map_upsampling_size);
			const unsigned int entry_upsampling_size = layer_derived->entry_upsampling_size;
			upsampling_sizes.push_back(entry_upsampling_size);
			const unsigned int upsampling_dimension_count = static_cast<unsigned int>(upsampling_sizes.size());
			const unsigned int spatial_dimension_count = static_cast<unsigned int>(output_dimension_sizes.size());
			std::vector<unsigned int> output_slices(upsampling_sizes.size());
			output_slices[0] = 1;
			for(unsigned int i = 0; i < upsampling_dimension_count - 1; ++i)
			{
				int dimension_size = (i < spatial_dimension_count) ? output_dimension_sizes[i] : output_configuration_specific.feature_map_count;
				output_slices[i + 1] = output_slices[i] * dimension_size;
			}
			unsigned int upsampling_elem_count = 1;
			for(unsigned int i = 0; i < upsampling_dimension_count; ++i)
				upsampling_elem_count *= upsampling_sizes[i];
			const unsigned int const_upsampling_elem_count = upsampling_elem_count;
			const unsigned int input_feature_map_count = input_configuration_specific_list[0].feature_map_count;

			std::vector<unsigned int> current_local_output_position(upsampling_dimension_count, 0);
			std::vector<unsigned int> offset_list(upsampling_elem_count);
			for(unsigned int i = 1; i < upsampling_elem_count; ++i)
			{
				int offset = 0;
				for(unsigned int j = 0; j < upsampling_dimension_count; ++j)
				{
					offset += static_cast<int>(output_slices[j]);
					if ((++current_local_output_position[j]) < upsampling_sizes[j])
					{
						offset_list[i] = offset_list[i-1] + offset;
						break;
					}
					current_local_output_position[j] = 0;
					offset -= static_cast<int>(upsampling_sizes[j] * output_slices[j]);
				}
			}

			const int total_workload = (entry_count / entry_upsampling_size) * input_configuration_specific_list[0].feature_map_count;
			const std::vector<unsigned int>::const_iterator dimension_sizes_it = input_dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator upsampling_sizes_it = upsampling_sizes.begin();
			const std::vector<unsigned int>::const_iterator output_slices_it = output_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_input_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int input_entry_id = workload_id / input_feature_map_count;
					int input_feature_map_id = workload_id - (input_entry_id * input_feature_map_count);

					const float * in_it_base = in_it_global + (input_entry_id * input_neuron_count) + (input_feature_map_id * input_neuron_count_per_feature_map);
					float * out_it_base = out_it_global + (input_entry_id * entry_upsampling_size * output_neuron_count ) + (input_feature_map_id * feature_map_upsampling_size * output_neuron_count_per_feature_map);

					std::fill_n(current_input_position.begin(), spatial_dimension_count, 0);
					for(const float * in_it = in_it_base; in_it != in_it_base + input_neuron_count_per_feature_map; ++in_it)
					{
						float val = *in_it;

						// Define the starting position of the first output elem
						float * out_it = out_it_base;
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
							out_it += current_input_position[i] * (*(upsampling_sizes_it + i)) * (*(output_slices_it + i));

						for(unsigned int i = 0; i < const_upsampling_elem_count; ++i)
							*(out_it + (*(offset_list_it + i))) = val;

						// Go to the next input element
						for(unsigned int i = 0; i < spatial_dimension_count; ++i)
						{
							if ((++current_input_position[i]) < *( dimension_sizes_it + i))
								break;
							current_input_position[i] = 0;
						}
					}
				}
			}
		}
	}
}
