/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "average_subsampling_layer_hessian_plain.h"

#include "../average_subsampling_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		const int average_subsampling_layer_hessian_plain::max_dimension_count = 4;

		average_subsampling_layer_hessian_plain::average_subsampling_layer_hessian_plain()
		{
		}

		average_subsampling_layer_hessian_plain::~average_subsampling_layer_hessian_plain()
		{
		}

		const boost::uuids::uuid& average_subsampling_layer_hessian_plain::get_uuid() const
		{
			return average_subsampling_layer::layer_guid;
		}

		void average_subsampling_layer_hessian_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const std::vector<float>::const_iterator in_it_global = input_buffer->begin();
			const std::vector<float>::iterator out_it_global = output_buffer->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);
			const std::vector<unsigned int>& subsampling_sizes = layer_derived->subsampling_sizes;
			const unsigned int dimension_count = static_cast<unsigned int>(layer_derived->subsampling_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];
			unsigned int subsampling_elem_count = 1;
			for(unsigned int i = 0; i < dimension_count; ++i)
				subsampling_elem_count *= subsampling_sizes[i];
			const unsigned int const_subsampling_elem_count = subsampling_elem_count;
			const float mult = 1.0F / static_cast<float>(subsampling_elem_count);
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;

			std::vector<unsigned int> current_local_input_position(dimension_count, 0);
			std::vector<unsigned int> offset_list(subsampling_elem_count);
			for(unsigned int i = 1; i < subsampling_elem_count; ++i)
			{
				int offset = 0;
				for(unsigned int j = 0; j < dimension_count; ++j)
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
			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator subsampling_sizes_it = subsampling_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / feature_map_count;
					int feature_map_id = workload_id - (entry_id * feature_map_count);

					std::vector<float>::const_iterator in_it_base = in_it_global + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
					std::vector<float>::iterator out_it_base = out_it_global + (entry_id * output_neuron_count) + (feature_map_id * output_neuron_count_per_feature_map);

					std::fill_n(current_output_position.begin(), dimension_count, 0);
					for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						// Define the starting position of the first input elem
						std::vector<float>::const_iterator in_it = in_it_base;
						for(unsigned int i = 0; i < dimension_count; ++i)
							in_it += current_output_position[i] * (*(subsampling_sizes_it + i)) * (*(input_slices_it + i));

						float sum = 0.0F;
						for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
						{
							sum += *(in_it + (*(offset_list_it + i)));
						}
						*out_it = sum * mult;

						// Go to the next output element
						for(unsigned int i = 0; i < dimension_count; ++i)
						{
							if ((++current_output_position[i]) < *( dimension_sizes_it + i))
								break;
							current_output_position[i] = 0;
						}
					}
				}
			}
		}

		void average_subsampling_layer_hessian_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const std::vector<float>::iterator in_err_it_global = input_errors->begin();
			const std::vector<float>::const_iterator out_err_it_global = output_errors->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);
			const std::vector<unsigned int>& subsampling_sizes = layer_derived->subsampling_sizes;
			const unsigned int dimension_count = static_cast<unsigned int>(layer_derived->subsampling_sizes.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];
			unsigned int subsampling_elem_count = 1;
			for(unsigned int i = 0; i < dimension_count; ++i)
				subsampling_elem_count *= subsampling_sizes[i];
			const unsigned int const_subsampling_elem_count = subsampling_elem_count;
			const float mult = 1.0F / static_cast<float>(subsampling_elem_count * subsampling_elem_count);
			const unsigned int feature_map_count = output_configuration_specific.feature_map_count;

			std::vector<unsigned int> current_local_input_position(dimension_count, 0);
			std::vector<unsigned int> offset_list(subsampling_elem_count);
			for(unsigned int i = 1; i < subsampling_elem_count; ++i)
			{
				int offset = 0;
				for(unsigned int j = 0; j < dimension_count; ++j)
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
			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const std::vector<unsigned int>::const_iterator subsampling_sizes_it = subsampling_sizes.begin();
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				nnforge_array<unsigned int, max_dimension_count> current_output_position;

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / feature_map_count;
					int feature_map_id = workload_id - (entry_id * feature_map_count);

					std::vector<float>::iterator in_err_it_base = in_err_it_global + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
					std::vector<float>::const_iterator out_err_it_base = out_err_it_global + (entry_id * output_neuron_count) + (feature_map_id * output_neuron_count_per_feature_map);

					std::fill_n(current_output_position.begin(), dimension_count, 0);
					std::fill_n(in_err_it_base, input_neuron_count_per_feature_map, 0.0F);
					for(std::vector<float>::const_iterator out_it = out_err_it_base; out_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_it)
					{
						// Define the starting position of the first input elem
						std::vector<float>::iterator in_it = in_err_it_base;
						for(unsigned int i = 0; i < dimension_count; ++i)
							in_it += current_output_position[i] * (*(subsampling_sizes_it + i)) * (*(input_slices_it + i));

						float err = *out_it * mult;
						for(unsigned int i = 0; i < const_subsampling_elem_count; ++i)
						{
							*(in_it + (*(offset_list_it + i))) = err;
						}

						// Go to the next output element
						for(unsigned int i = 0; i < dimension_count; ++i)
						{
							if ((++current_output_position[i]) < *( dimension_sizes_it + i))
								break;
							current_output_position[i] = 0;
						}
					}
				}
			}
		}

		bool average_subsampling_layer_hessian_plain::is_in_place_backprop() const
		{
			return false;
		}
	}
}
