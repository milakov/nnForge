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

#include "convolution_layer_updater_plain.h"

#include "../convolution_layer.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int convolution_layer_updater_plain::max_dimension_count = 4;

		convolution_layer_updater_plain::convolution_layer_updater_plain()
		{
		}

		convolution_layer_updater_plain::~convolution_layer_updater_plain()
		{
		}

		std::string convolution_layer_updater_plain::get_type_name() const
		{
			return convolution_layer::layer_type_name;
		}

		void convolution_layer_updater_plain::run_forward_propagation(
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
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			std::vector<unsigned int> window_sizes_extended = layer_derived->window_sizes;
			window_sizes_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& window_sizes = window_sizes_extended;

			std::vector<unsigned int> strides_extended = layer_derived->strides;
			strides_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& strides = strides_extended;

			std::vector<unsigned int> left_zero_padding_extended = layer_derived->left_zero_padding;
			left_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& left_zero_padding = left_zero_padding_extended;

			std::vector<unsigned int> right_zero_padding_extended = layer_derived->right_zero_padding;
			right_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& right_zero_padding = right_zero_padding_extended;

			std::vector<unsigned int> input_dimension_sizes_extended = input_configuration_specific_list[0].dimension_sizes;
			input_dimension_sizes_extended .resize(max_dimension_count, 1);
			const std::vector<unsigned int>& input_dimension_sizes = input_dimension_sizes_extended;

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
			const std::vector<unsigned int>::const_iterator strides_it = strides.begin();

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

					std::fill_n(current_input_position.begin(), max_dimension_count, 0);
					std::fill_n(current_output_position.begin(), max_dimension_count, 0);
					for(float * out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						float sum = *(biases + output_feature_map_id);
						std::vector<float>::const_iterator weights_it = weights + (output_feature_map_id * (const_window_elem_count * input_feature_map_count));
						int in_it_offset2 = 0;

						for(unsigned int i = 0; i < dimension_count; ++i)
							current_input_position[i] = static_cast<int>(current_output_position[i] * strides_it[i]) - static_cast<int>(left_zero_padding[i]);

						for(unsigned int i = 0; i < dimension_count; ++i)
							in_it_offset2 += current_input_position[i] * (*(input_slices_it + i));

						for(unsigned int input_feature_map_id = 0; input_feature_map_id < input_feature_map_count; ++input_feature_map_id)
						{
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

		void convolution_layer_updater_plain::run_backward_data_propagation(
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
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			std::vector<unsigned int> window_sizes_extended = layer_derived->window_sizes;
			window_sizes_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& window_sizes = window_sizes_extended;

			std::vector<unsigned int> strides_extended = layer_derived->strides;
			strides_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& strides = strides_extended;

			std::vector<unsigned int> left_zero_padding_extended = layer_derived->left_zero_padding;
			left_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& left_zero_padding = left_zero_padding_extended;

			std::vector<unsigned int> right_zero_padding_extended = layer_derived->right_zero_padding;
			right_zero_padding_extended.resize(max_dimension_count, 0);
			const std::vector<unsigned int>& right_zero_padding = right_zero_padding_extended;

			std::vector<unsigned int> input_dimension_sizes_extended = input_configuration_specific_list[0].dimension_sizes;
			input_dimension_sizes_extended .resize(max_dimension_count, 1);
			const std::vector<unsigned int>& input_dimension_sizes = input_dimension_sizes_extended;

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
			const int total_workload = entry_count * input_feature_map_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();
			const std::vector<unsigned int>::const_iterator strides_it = strides.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(window_sizes,left_zero_padding,right_zero_padding,input_dimension_sizes)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;
				nnforge_array<int, max_dimension_count> current_input_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / input_feature_map_count;
					int input_feature_map_id = workload_id - (entry_id * input_feature_map_count);

					const float * out_err_it_base = out_err_it_global + (entry_id * output_neuron_count);
					float * in_err_it_base = in_err_it_global + (entry_id * input_neuron_count) + (input_feature_map_id * input_neuron_count_per_feature_map);
					std::vector<float>::const_iterator weights_it_base = weights + (const_window_elem_count * input_feature_map_id);

					if (!add_update_to_destination)
						std::fill_n(in_err_it_base, input_neuron_count_per_feature_map, 0.0F);

					std::fill_n(current_input_position.begin(), max_dimension_count, 0);
					std::fill_n(current_output_position.begin(), max_dimension_count, 0);
					for(const float * out_err_it_base2 = out_err_it_base; out_err_it_base2 != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it_base2)
					{
						int in_err_offset = 0;

						for(unsigned int i = 0; i < dimension_count; ++i)
							current_input_position[i] = static_cast<int>(current_output_position[i] * strides_it[i]) - static_cast<int>(left_zero_padding[i]);

						for(unsigned int i = 0; i < dimension_count; ++i)
							in_err_offset += current_input_position[i] * (*(input_slices_it + i));

						for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_feature_map_count; ++output_feature_map_id)
						{
							const float * out_err_it = out_err_it_base2 + (output_feature_map_id * output_neuron_count_per_feature_map);
							std::vector<float>::const_iterator weights_it_base2 = weights_it_base + (output_feature_map_id * (const_window_elem_count * input_feature_map_count));
							std::vector<float>::const_iterator weights_it = weights_it_base2;
							float current_err = *out_err_it;

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
											{
												float w = *weights_it;
												*(in_err_it_base + (in_err_offset + *(offset_list_it + ind))) += (w * current_err);
											}
											++ind;
											++weights_it;
										}
									}
								}
							}
						}

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

		void convolution_layer_updater_plain::run_backward_weights_propagation(
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
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const float * const in_it_global = *input_neurons_buffers[0];
			const float * const out_err_it_global = *output_errors_buffer;
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			std::vector<unsigned int> window_sizes_extended = layer_derived->window_sizes;
			window_sizes_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& window_sizes = window_sizes_extended;

			std::vector<unsigned int> strides_extended = layer_derived->strides;
			strides_extended.resize(max_dimension_count, 1);
			const std::vector<unsigned int>& strides = strides_extended;

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

			const std::vector<float>::iterator gradient_weights = (*gradient)[0].begin();
			const std::vector<float>::iterator gradient_biases = (*gradient)[1].begin();

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
			const int total_workload = output_feature_map_count * input_feature_map_count;
			const unsigned int const_entry_count = entry_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();
			const std::vector<unsigned int>::const_iterator strides_it = strides.begin();
			const int const_updater_count = entry_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(window_sizes,left_zero_padding,right_zero_padding,input_dimension_sizes)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;
				nnforge_array<int, max_dimension_count> current_input_position;
				std::vector<float> weights_local(const_window_elem_count, 0.0F);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int feature_map_pair_id = workload_id;
					int output_feature_map_id = feature_map_pair_id / input_feature_map_count;
					int input_feature_map_id = feature_map_pair_id - (output_feature_map_id * input_feature_map_count);

					std::vector<float>::iterator gradient_weights_it_base = gradient_weights + (output_feature_map_id * (const_window_elem_count * input_feature_map_count)) + (const_window_elem_count * input_feature_map_id);
					std::fill_n(weights_local.begin(), const_window_elem_count, 0.0F);

					for(int entry_id = 0; entry_id < const_updater_count; ++entry_id)
					{
						const float * in_it_base = in_it_global + (entry_id * input_neuron_count) + (input_feature_map_id * input_neuron_count_per_feature_map);
						const float * out_err_it_base = out_err_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

						std::fill_n(current_input_position.begin(), max_dimension_count, 0);
						std::fill_n(current_output_position.begin(), max_dimension_count, 0);
						for(const float * out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it)
						{
							int in_it_offset = 0;

							for(unsigned int i = 0; i < dimension_count; ++i)
								current_input_position[i] = static_cast<int>(current_output_position[i] * strides_it[i]) - static_cast<int>(left_zero_padding[i]);

							for(unsigned int i = 0; i < dimension_count; ++i)
								in_it_offset += current_input_position[i] * (*(input_slices_it + i));

							float current_err = *out_err_it;

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
											{
												float in_neuron = *(in_it_base + (in_it_offset + *(offset_list_it + ind)));
												weights_local[ind] += (in_neuron * current_err);
											}
											++ind;
										}
									}
								}
							}

							// Go to the next output element
							for(unsigned int i = 0; i < dimension_count; ++i)
							{
								if ((++current_output_position[i]) < *(output_dimension_sizes_it + i))
									break;
								current_output_position[i] = 0;
							}
						}
					}

					std::vector<float>::iterator weights_local_it = weights_local.begin();
					for(std::vector<float>::iterator it = gradient_weights_it_base; it != gradient_weights_it_base + const_window_elem_count; ++it, ++weights_local_it)
						*it += *weights_local_it;
				}
			}

			const int total_workload_bias = output_feature_map_count;
			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload_bias; ++workload_id)
			{
				int output_feature_map_id = workload_id;

				float sum = 0.0F;
				for(int entry_id = 0; entry_id < const_updater_count; ++entry_id)
				{
					float local_sum = 0.0F;
					const float * out_err_it_base = out_err_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					for(const float * out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it)
						local_sum += *out_err_it;

					sum += local_sum;
				}

				*(gradient_biases + output_feature_map_id) += sum;
			}
		}

		bool convolution_layer_updater_plain::is_backward_data_dependent_on_input_buffer(
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

		bool convolution_layer_updater_plain::is_backward_data_dependent_on_output_buffer(
			unsigned int action_input_index,
			const std::set<layer_action>& actions,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return false;
		}

		bool convolution_layer_updater_plain::is_backward_weights_dependent_on_input_buffer(
			unsigned int data_input_index,
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
