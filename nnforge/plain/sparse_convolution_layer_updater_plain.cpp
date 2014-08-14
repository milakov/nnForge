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

#include "sparse_convolution_layer_updater_plain.h"

#include "../sparse_convolution_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int sparse_convolution_layer_updater_plain::max_dimension_count = 4;

		sparse_convolution_layer_updater_plain::sparse_convolution_layer_updater_plain()
		{
		}

		sparse_convolution_layer_updater_plain::~sparse_convolution_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& sparse_convolution_layer_updater_plain::get_uuid() const
		{
			return sparse_convolution_layer::layer_guid;
		}

		void sparse_convolution_layer_updater_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const std::vector<float>::const_iterator in_it_global = input_buffer->begin() + input_neuron_count * offset_input_entry_id;
			const std::vector<float>::iterator out_it_global = output_buffer->begin();
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);
			const std::vector<unsigned int>& window_sizes = layer_derived->window_sizes;
			const unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];
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
			const unsigned int input_feature_map_count = input_configuration_specific.feature_map_count;
			const int total_workload = updater_count * output_feature_map_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					std::vector<float>::iterator out_it_base = out_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					std::vector<float>::const_iterator in_it_base = in_it_global + entry_id * input_neuron_count;

					const int start_column_index = row_indices[output_feature_map_id];
					const int end_column_index = row_indices[output_feature_map_id + 1];

					std::fill_n(current_output_position.begin(), dimension_count, 0);
					for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						float sum = *(biases + output_feature_map_id);
						std::vector<float>::const_iterator weights_it = weights + start_column_index * const_window_elem_count;
						std::vector<float>::const_iterator in_it_base2 = in_it_base;
						for(unsigned int i = 0; i < dimension_count; ++i)
							in_it_base2 += current_output_position[i] * (*(input_slices_it + i));

						for(int column_index = start_column_index; column_index < end_column_index; ++column_index)
						{
							int input_feature_map_id = column_indices[column_index];

							// Define the starting position of the first input elem
							std::vector<float>::const_iterator in_it = in_it_base2 + (input_feature_map_id * input_neuron_count_per_feature_map);

							for(unsigned int i = 0; i < const_window_elem_count; ++i)
							{
								sum += (*(in_it + *(offset_list_it + i))) * (*weights_it);
								++weights_it;
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

		void sparse_convolution_layer_updater_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count) const
		{
			const std::vector<float>::iterator in_err_it_global = input_errors->begin();
			const std::vector<float>::const_iterator out_err_it_global = output_errors->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);
			const std::vector<unsigned int>& window_sizes = layer_derived->window_sizes;
			const unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];
			unsigned int window_elem_count = 1;
			for(unsigned int i = 0; i < dimension_count; ++i)
				window_elem_count *= window_sizes[i];
			const unsigned int const_window_elem_count = window_elem_count;

			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			const std::vector<int>::const_iterator column_indices = (*data_custom)[0].begin();
			const std::vector<int>::const_iterator row_indices = (*data_custom)[1].begin();

			std::vector<std::vector<std::pair<int, int> > > in_fm_out_fm_weight_pos_list_list(input_configuration_specific.feature_map_count);
			for(int output_feature_map_id = 0; output_feature_map_id < output_configuration_specific.feature_map_count; ++output_feature_map_id)
			{
				const int start_column_index = row_indices[output_feature_map_id];
				const int end_column_index = row_indices[output_feature_map_id + 1];
				for(int column_index = start_column_index; column_index < end_column_index; ++column_index)
				{
					int input_feature_map_id = column_indices[column_index];
					in_fm_out_fm_weight_pos_list_list[input_feature_map_id].push_back(std::make_pair(output_feature_map_id, column_index));
				}
			}

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
			const unsigned int input_feature_map_count = input_configuration_specific.feature_map_count;
			const int total_workload = updater_count * input_feature_map_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();
			const std::vector<std::vector<std::pair<int, int> > >::const_iterator in_fm_out_fm_weight_pos_it = in_fm_out_fm_weight_pos_list_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / input_feature_map_count;
					int input_feature_map_id = workload_id - (entry_id * input_feature_map_count);

					std::vector<float>::const_iterator out_err_it_base = out_err_it_global + (entry_id * output_neuron_count);
					std::vector<float>::iterator in_err_it_base = in_err_it_global + (entry_id * input_neuron_count) + (input_feature_map_id * input_neuron_count_per_feature_map);
					const std::vector<std::pair<int, int> >& out_fm_weight_pos_list = in_fm_out_fm_weight_pos_it[input_feature_map_id];

					std::fill_n(in_err_it_base, input_neuron_count_per_feature_map, 0.0F);
					std::fill_n(current_output_position.begin(), dimension_count, 0);
					for(std::vector<float>::const_iterator out_err_it_base2 = out_err_it_base; out_err_it_base2 != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it_base2)
					{
						std::vector<float>::iterator in_err_it = in_err_it_base;
						for(unsigned int i = 0; i < dimension_count; ++i)
							in_err_it += current_output_position[i] * (*(input_slices_it + i));

						for(std::vector<std::pair<int, int> >::const_iterator it = out_fm_weight_pos_list.begin(); it != out_fm_weight_pos_list.end(); ++it)
						{
							int output_feature_map_id = it->first;
							int weight_block_id = it->second;

							std::vector<float>::const_iterator out_err_it = out_err_it_base2 + (output_feature_map_id * output_neuron_count_per_feature_map);
							std::vector<float>::const_iterator weights_it = weights + weight_block_id * const_window_elem_count;
							float current_err = *out_err_it;
							for(unsigned int i = 0; i < const_window_elem_count; ++i)
							{
								float w = *weights_it;
								*(in_err_it + *(offset_list_it + i)) += (w * current_err);
								++weights_it;
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

		void sparse_convolution_layer_updater_plain::update_weights(
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			layer_data_smart_ptr gradient,
			const_layer_data_custom_smart_ptr data_custom,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const std::vector<float>::const_iterator in_it_global = input_neurons->begin() + input_neuron_count * offset_input_entry_id;
			const std::vector<float>::const_iterator out_err_it_global = output_errors->begin();
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);
			const std::vector<unsigned int>& window_sizes = layer_derived->window_sizes;
			unsigned int feature_map_connection_count = layer_derived->feature_map_connection_count;
			const unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];
			unsigned int window_elem_count = 1;
			for(unsigned int i = 0; i < dimension_count; ++i)
				window_elem_count *= window_sizes[i];
			const unsigned int const_window_elem_count = window_elem_count;

			const std::vector<float>::iterator gradient_weights = (*gradient)[0].begin();
			const std::vector<float>::iterator gradient_biases = (*gradient)[1].begin();

			const std::vector<int>::const_iterator column_indices = (*data_custom)[0].begin();
			const std::vector<int>::const_iterator row_indices = (*data_custom)[1].begin();

			std::vector<std::pair<int, int> > out_fm_in_fm_list(feature_map_connection_count);
			int i = 0;
			for(int output_feature_map_id = 0; output_feature_map_id < output_configuration_specific.feature_map_count; ++output_feature_map_id)
			{
				const int start_column_index = row_indices[output_feature_map_id];
				const int end_column_index = row_indices[output_feature_map_id + 1];
				for(int column_index = start_column_index; column_index < end_column_index; ++column_index)
				{
					int input_feature_map_id = column_indices[column_index];
					out_fm_in_fm_list[i].first = output_feature_map_id;
					out_fm_in_fm_list[i].second = input_feature_map_id;
					++i;
				}
			}

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
			const unsigned int input_feature_map_count = input_configuration_specific.feature_map_count;
			const int total_workload = feature_map_connection_count;
			const unsigned int const_entry_count = updater_count;
			const std::vector<unsigned int>::const_iterator output_dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();
			const std::vector<std::pair<int, int> >::const_iterator out_fm_in_fm_it = out_fm_in_fm_list.begin();
			const int const_updater_count = updater_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;
				std::vector<float> weights_local(const_window_elem_count, 0.0F);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int weight_block_id = workload_id;
					int output_feature_map_id = out_fm_in_fm_it[weight_block_id].first;
					int input_feature_map_id = out_fm_in_fm_it[weight_block_id].second;

					std::fill_n(weights_local.begin(), const_window_elem_count, 0.0F);

					for(int entry_id = 0; entry_id < const_updater_count; ++entry_id)
					{
						std::vector<float>::const_iterator in_it_base = in_it_global + (entry_id * input_neuron_count) + (input_feature_map_id * input_neuron_count_per_feature_map);
						std::vector<float>::const_iterator out_err_it_base = out_err_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);

						std::fill_n(current_output_position.begin(), dimension_count, 0);
						for(std::vector<float>::const_iterator out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it)
						{
							std::vector<float>::const_iterator in_it = in_it_base;
							for(unsigned int i = 0; i < dimension_count; ++i)
								in_it += current_output_position[i] * (*(input_slices_it + i));

							float current_err = *out_err_it;
							for(unsigned int i = 0; i < const_window_elem_count; ++i)
							{
								float in_neuron = *(in_it + *(offset_list_it + i));
								weights_local[i] += (in_neuron * current_err);
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

					std::vector<float>::iterator gradient_weights_it_base = gradient_weights + weight_block_id * const_window_elem_count;
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
					std::vector<float>::const_iterator out_err_it_base = out_err_it_global + (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					for(std::vector<float>::const_iterator out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it)
						sum += *out_err_it;
				}

				*(gradient_biases + output_feature_map_id) += sum;
			}
		}

		bool sparse_convolution_layer_updater_plain::is_in_place_backprop() const
		{
			return false;
		}
	}
}
