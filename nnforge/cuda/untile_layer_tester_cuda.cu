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

#include "untile_layer_tester_cuda.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../untile_layer.h"
#include "../nn_types.h"

__global__ void untile_kernel(
	float * __restrict output,
	const float * __restrict input,
	const int * __restrict output_positions,
	const int * __restrict output_offsets,
	int neuron_count_per_input_feature_map,
	int neuron_count_per_output_feature_map,
	int feature_map_count,
	int output_entry_count,
	int_fastdiv local_entry_count)
{
	int input_neuron_output_local_entry_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int output_entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	int input_neuron_id = input_neuron_output_local_entry_pair_id / local_entry_count;
	int local_entry_id = input_neuron_output_local_entry_pair_id - input_neuron_id * local_entry_count;

	bool b_valid = (input_neuron_id < neuron_count_per_input_feature_map) && (local_entry_id < local_entry_count) && (feature_map_id < feature_map_count) && (output_entry_id < output_entry_count);
	if (b_valid)
	{
		int input_entry_id = output_entry_id * local_entry_count + local_entry_id;
		int input_offset = (input_entry_id * feature_map_count + feature_map_id) * neuron_count_per_input_feature_map + input_neuron_id;
		int output_neuron_offset = __load_nc(output_positions + input_neuron_id) + __load_nc(output_offsets + local_entry_id);
		float val = __load_nc(input + input_offset);
		int output_offset = (output_entry_id * feature_map_count + feature_map_id) * neuron_count_per_output_feature_map + output_neuron_offset;
		output[output_offset] = val;
	}
}

namespace nnforge
{
	namespace cuda
	{
		untile_layer_tester_cuda::untile_layer_tester_cuda()
		{
		}

		untile_layer_tester_cuda::~untile_layer_tester_cuda()
		{
		}

		void untile_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			const float * input = *input_buffer;
			float * output = *additional_buffers[0];
			const int * output_positions = *additional_buffers[1];
			const int * output_offsets = *additional_buffers[2];

			if (entry_count % total_tiling_factor != 0)
				throw neural_network_exception((boost::format("untile_layer_tester_cuda: entry_count (%1%) is not evenly divisible by total_tiling_factor (%2%)") % entry_count % (int)total_tiling_factor).str());
			int output_entry_count = entry_count / total_tiling_factor;

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map * total_tiling_factor,
				output_configuration_specific.feature_map_count,
				output_entry_count);

			untile_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				output,
				input,
				output_positions,
				output_offsets,
				input_elem_count_per_feature_map,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				output_entry_count,
				total_tiling_factor);
		}

		std::vector<size_t> untile_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back((output_elem_count_per_entry * sizeof(float) + total_tiling_factor - 1) / (int)total_tiling_factor);

			return res;
		}

		std::vector<size_t> untile_layer_tester_cuda::get_sizes_of_additional_buffers_fixed() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_feature_map * sizeof(int));
			res.push_back(total_tiling_factor * sizeof(int));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr untile_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		void untile_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const untile_layer> layer_derived = nnforge_dynamic_pointer_cast<const untile_layer>(layer_schema);

			upsampling_sizes_list = layer_derived->upsampling_sizes_list;
			total_tiling_factor = layer_derived->get_tiling_factor().get_inverse();
		}

		void untile_layer_tester_cuda::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
			{
				std::vector<int> position_list(input_elem_count_per_feature_map);
				{
					std::vector<unsigned int> tiling_sizes(input_configuration_specific.dimension_sizes.size(), 1);
					for(int i = 0; i < upsampling_sizes_list.size(); ++i)
					{
						const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
						for(int j = 0; j < upsampling_sizes.size(); ++j)
							tiling_sizes[j] *= upsampling_sizes[j];
					}

					std::vector<unsigned int> spatial_pos(input_configuration_specific.dimension_sizes.size(), 0);
					for(unsigned int i = 0; i < input_elem_count_per_feature_map; ++i)
					{
						int pos = spatial_pos.back() * tiling_sizes.back();
						for(int j = static_cast<int>(spatial_pos.size() - 2); j >= 0; --j)
							pos = pos * output_configuration_specific.dimension_sizes[j] + spatial_pos[j] * tiling_sizes[j];
						position_list[i] = pos;

						for(int j = 0; j < spatial_pos.size(); ++j)
						{
							if ((++spatial_pos[j]) < input_configuration_specific.dimension_sizes[j])
								break;
							spatial_pos[j] = 0;
						}
					}
				}
				cuda_safe_call(cudaMemcpy(*additional_buffers[1], &(*position_list.begin()), sizeof(int) * position_list.size(), cudaMemcpyHostToDevice));
			}

			{
				std::vector<int> offset_list(total_tiling_factor);
				{
					std::vector<std::vector<unsigned int> > positions_list;
					positions_list.push_back(std::vector<unsigned int>(output_configuration_specific.dimension_sizes.size(), 0));

					std::vector<unsigned int> total_upsampling_sizes(upsampling_sizes_list.front().size(), 1);

					for(int level = static_cast<unsigned int>(upsampling_sizes_list.size()) - 1; level >= 0; --level)
					{
						std::vector<std::vector<unsigned int> > new_positions_list;
						const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[level];

						unsigned int local_tiling_count = 1;
						for(std::vector<unsigned int>::const_iterator it = upsampling_sizes.begin(); it != upsampling_sizes.end(); ++it)
							local_tiling_count *= *it;

						for(std::vector<std::vector<unsigned int> >::const_iterator it = positions_list.begin(); it != positions_list.end(); ++it)
						{
							const std::vector<unsigned int>& current_positions = *it;

							std::vector<unsigned int> local_pos(upsampling_sizes.size(), 0);
							for(unsigned int i = 0; i < local_tiling_count; ++i)
							{
								std::vector<unsigned int> new_untiled_positions(current_positions);
								for(unsigned int j = 0; j < static_cast<unsigned int>(upsampling_sizes.size()); ++j)
									new_untiled_positions[j] += local_pos[j] * total_upsampling_sizes[j];

								new_positions_list.push_back(new_untiled_positions);

								for(int j = 0; j < local_pos.size(); ++j)
								{
									if ((++local_pos[j]) < upsampling_sizes[j])
										break;
									local_pos[j] = 0;
								}
							}
						}

						for(unsigned int i = 0; i < static_cast<unsigned int>(total_upsampling_sizes.size()); ++i)
							total_upsampling_sizes[i] *= upsampling_sizes[i];

						positions_list = new_positions_list;
					}

					for(unsigned int i = 0; i < total_tiling_factor; ++i)
					{
						const std::vector<unsigned int>& positions = positions_list[i];
						int pos = positions.back();
						for(int j = static_cast<int>(positions.size() - 2); j >= 0; --j)
							pos = pos * output_configuration_specific.dimension_sizes[j] + positions[j];
						offset_list[i] = pos;
					}
				}
				cuda_safe_call(cudaMemcpy(*additional_buffers[2], &(*offset_list.begin()), sizeof(int) * offset_list.size(), cudaMemcpyHostToDevice));
			}
		}
	}
}
