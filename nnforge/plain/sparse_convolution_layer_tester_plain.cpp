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

#include "sparse_convolution_layer_tester_plain.h"

#include "../sparse_convolution_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int sparse_convolution_layer_tester_plain::max_dimension_count = 4;

		sparse_convolution_layer_tester_plain::sparse_convolution_layer_tester_plain()
		{
		}

		sparse_convolution_layer_tester_plain::~sparse_convolution_layer_tester_plain()
		{
		}

		std::string sparse_convolution_layer_tester_plain::get_type_name() const
		{
			return sparse_convolution_layer::layer_type_name;
		}

		void sparse_convolution_layer_tester_plain::run_forward_propagation(
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
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			std::vector<unsigned int> window_sizes_extended = layer_derived->window_sizes;
			window_sizes_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& window_sizes = window_sizes_extended;

			std::vector<unsigned int> left_zero_padding_extended = layer_derived->left_zero_padding;
			left_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& left_zero_padding = left_zero_padding_extended;

			std::vector<unsigned int> right_zero_padding_extended = layer_derived->right_zero_padding;
			right_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& right_zero_padding = right_zero_padding_extended;

			std::vector<unsigned int> input_dimension_sizes_extended = input_configuration_specific_list[0].dimension_sizes;
			input_dimension_sizes_extended .resize(max_dimension_count, 1);
			const std::vector<unsigned int>& input_dimension_sizes = input_dimension_sizes_extended ;

			const unsigned int dimension_count = static_cast<unsigned int>(layer_derived->window_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific_list[0].dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific_list[0].dimension_sizes[i];
			unsigned int window_elem_count = 1;
			for(unsigned int i = 0; i < dimension_count; ++i)
				window_elem_count *= window_sizes[i];
			const unsigned int const_window_elem_count = window_elem_count;

			const std::vector<float>::const_iterator weights = (*data)[0].begin();
			const std::vector<float>::const_iterator biases = (*data)[1].begin();

			const std::vector<int>::const_iterator column_indices = (*data_custom)[0].begin();
			const std::vector<int>::const_iterator row_indices = (*data_custom)[1].begin();

			std::vector<unsigned int> current_local_input_position(dimension_count, 0);
			std::vector<unsigned int> offset_list(window_elem_count);
			for(unsigned int i = 1; i < window_elem_count; ++i)
			{
				int offset = 0;
				for(unsigned int j = 0; j < dimension_count; ++j)
				{
					offset += static_cast<int>(input_slices[j]);
					if ((++current_local_input_position[j]) < window_sizes[j])
					{
						offset_list[i] = offset_list[i-1] + offset;
						break;
					}
					current_local_input_position[j] = 0;
					offset -= static_cast<int>(window_sizes[j] * input_slices[j]);
				}
			}

			const unsigned int output_feature_map_count = output_configuration_specific.feature_map_count;
			const unsigned int input_feature_map_count = input_configuration_specific_list[0].feature_map_count;
			const int total_workload = entry_count * output_feature_map_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(window_sizes,left_zero_padding,right_zero_padding,input_dimension_sizes)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;
				nnforge_array<int, max_dimension_count> current_input_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					float * out_it_base = out_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					const float * in_it_base = in_it_global + entry_id * input_neuron_count;

					const int start_column_index = row_indices[output_feature_map_id];
					const int end_column_index = row_indices[output_feature_map_id + 1];

					std::fill_n(current_input_position.begin(), max_dimension_count, 0);
					std::fill_n(current_output_position.begin(), max_dimension_count, 0);
					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						float sum = *(biases + output_feature_map_id);
						std::vector<float>::const_iterator weights_it = weights + start_column_index * const_window_elem_count;

						int in_it_offset2 = 0;

						for(unsigned int i = 0; i < dimension_count; ++i)
							current_input_position[i] = static_cast<int>(current_output_position[i]) - static_cast<int>(left_zero_padding[i]);

						for(unsigned int i = 0; i < dimension_count; ++i)
							in_it_offset2 += current_input_position[i] * (*(input_slices_it + i));

						for(int column_index = start_column_index; column_index < end_column_index; ++column_index)
						{
							int input_feature_map_id = column_indices[column_index];

							// Define the starting position of the first input elem
							int in_it_offset = in_it_offset2 + (input_feature_map_id * input_neuron_count_per_feature_map);

							int ind = 0;
							for(int w = current_input_position[3]; w < current_input_position[3] + static_cast<int>(window_sizes[3]); ++w)
							{
								bool fit3 = ((unsigned int)w < (unsigned int)input_dimension_sizes[3]);
								for(int z = current_input_position[2]; z < current_input_position[2] + static_cast<int>(window_sizes[2]); ++z)
								{
									bool fit2 = fit3 && ((unsigned int)z < (unsigned int)input_dimension_sizes[2]);
									for(int y = current_input_position[1]; y < current_input_position[1] + static_cast<int>(window_sizes[1]); ++y)
									{
										bool fit1 = fit2 && ((unsigned int)y < (unsigned int)input_dimension_sizes[1]);
										for(int x = current_input_position[0]; x < current_input_position[0] + static_cast<int>(window_sizes[0]); ++x)
										{
											bool fit0 = fit1 && ((unsigned int)x < (unsigned int)input_dimension_sizes[0]);
											if (fit0)
												sum += (*(in_it_base + (in_it_offset + *(offset_list_it + ind)))) * (*weights_it);
											++ind;
											++weights_it;
										}
									}
								}
							}
						}
						*out_it = sum;

						// Go to the next output element
						for(unsigned int i = 0; i < dimension_count; ++i)
						{
							if ((++current_output_position[i]) < *(output_dimension_sizes_it + i))
								break;
							current_output_position[i] = 0;
						}
					}
				}
			}
		}
	}
}
