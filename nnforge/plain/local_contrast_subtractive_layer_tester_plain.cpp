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

#include "local_contrast_subtractive_layer_tester_plain.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../local_contrast_subtractive_layer.h"

#include <cstring>

namespace nnforge
{
	namespace plain
	{
		std::string local_contrast_subtractive_layer_tester_plain::get_type_name() const
		{
			return local_contrast_subtractive_layer::layer_type_name;
		}

		void local_contrast_subtractive_layer_tester_plain::run_forward_propagation(
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
			const unsigned int neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			std::shared_ptr<const local_contrast_subtractive_layer> layer_derived = std::dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);
			const std::vector<std::vector<float> >& window_weights_list = layer_derived->window_weights_list;
			const std::vector<unsigned int>& feature_maps_affected = layer_derived->feature_maps_affected;
			const unsigned int dimension_count = static_cast<unsigned int>(window_weights_list.size());
			std::vector<unsigned int> input_slices(input_configuration_specific_list[0].dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific_list[0].dimension_sizes[i];

			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const unsigned int feature_maps_affected_count = static_cast<unsigned int>(feature_maps_affected.size());
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator feature_maps_affected_it = feature_maps_affected.begin();
			const float * const input_buffer_it = *input_buffers[0];
			float * const output_buffer_it = *output_buffer;
			const std::vector<std::vector<float> >::const_iterator window_weights_list_it = window_weights_list.begin();
			float * const working_buffer_it = *temporary_working_fixed_buffer;

			if ((feature_maps_affected_count != output_configuration_specific.feature_map_count) && (output_buffer_it != input_buffer_it))
				memcpy(output_buffer_it, input_buffer_it, output_configuration_specific.get_neuron_count() * entry_count * sizeof(float));

			const int total_workload = entry_count * feature_maps_affected_count;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) num_threads(openmp_thread_count)
			{
				std::vector<float *> local_additional_buffers;
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				local_additional_buffers.push_back(working_buffer_it + thread_id * neuron_count_per_feature_map);
				if (dimension_count > 1)
					local_additional_buffers.push_back(working_buffer_it + (openmp_thread_count + thread_id) * neuron_count_per_feature_map);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / feature_maps_affected_count;
					int affected_feature_map_id = workload_id - (entry_id * feature_maps_affected_count);

					unsigned int current_output_buffer_index = 0;
					unsigned int feature_map_id = *(feature_maps_affected_it + affected_feature_map_id);
					for(unsigned int dimension_id = 0; dimension_id < dimension_count; ++dimension_id)
					{
						float * out_it_base = local_additional_buffers[current_output_buffer_index];
						const float * in_it;
						if (dimension_id > 0)
							in_it = local_additional_buffers[1 - current_output_buffer_index];
						else
							in_it = input_buffer_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
						int max_output_size = *(dimension_sizes_it + dimension_id);
						int input_slice_size = *(input_slices_it + dimension_id);

						std::vector<unsigned int> current_output_position(dimension_count, 0);
						for(float * out_it = out_it_base; out_it != out_it_base + neuron_count_per_feature_map; ++out_it, ++in_it)
						{
							const std::vector<float>& current_window_weights_list = *(window_weights_list_it + dimension_id);
							float sum = *in_it * current_window_weights_list[0];

							int current_position = static_cast<int>(current_output_position[dimension_id]);
							int dest_forward = current_position;
							int dest_backward = dest_forward;
							for (std::vector<float>::const_iterator it = current_window_weights_list.begin() + 1; it != current_window_weights_list.end(); ++it)
							{
								dest_forward++;
								dest_backward--;
								int dest_forward_actual = (dest_forward < max_output_size) ? dest_forward : (((max_output_size << 1) - 1) - dest_forward);
								int dest_backward_actual = (dest_backward >= 0) ? dest_backward : (-1 - dest_backward);
								int offset_forward = ((dest_forward_actual - current_position) * input_slice_size);
								int offset_backward = ((dest_backward_actual - current_position) * input_slice_size);
								sum += (*(in_it + offset_forward) + *(in_it + offset_backward)) * (*it);
							}

							*out_it = sum;

							// Go to the next output element
							for(unsigned int i = 0; i < dimension_count; ++i)
							{
								if ((++current_output_position[i]) < *(dimension_sizes_it + i))
									break;
								current_output_position[i] = 0;
							}
						}

						current_output_buffer_index = 1 - current_output_buffer_index;
					} // for(unsigned int dimension_id

					// Subtract the gaussian blur
					{
						float * out_it = output_buffer_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
						const float * orig_it = input_buffer_it + (entry_id * neuron_count) + (feature_map_id * neuron_count_per_feature_map);
						const float * in_it = local_additional_buffers[1 - current_output_buffer_index];
						for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
							*(out_it + i) = *(orig_it + i) - *(in_it + i);
					}
				}
			} // #pragma parallel
		}

		int local_contrast_subtractive_layer_tester_plain::get_input_index_layer_can_write(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}

		size_t local_contrast_subtractive_layer_tester_plain::get_temporary_working_fixed_buffer_size(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			unsigned int elem_count_per_intermediate_elem = output_configuration_specific.get_neuron_count_per_feature_map();
			return elem_count_per_intermediate_elem * plain_config->openmp_thread_count * (output_configuration_specific.dimension_sizes.size() > 1 ? 2 : 1) * sizeof(float);
		}
	}
}
