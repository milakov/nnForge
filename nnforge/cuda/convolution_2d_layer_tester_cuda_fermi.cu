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

#include "convolution_2d_layer_tester_cuda_fermi.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "space_filling_curve.h"

#include "../convolution_layer.h"
#include "../nn_types.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;

#define FEATURE_MAP_BLOCK_SIZE 4

namespace nnforge
{
	namespace cuda
	{
		template<int BLOCK_SIZE>
		__global__ void convolution_2d_tex_blocked_kernel_fermi(
			float * __restrict output,
			const float * __restrict weights,
			const float * __restrict biases,
			const packed_config<2> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int input_width,
			int input_height,
			int window_width,
			int window_height,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < output_width) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = window_width * window_height * input_feature_map_count;
				packed_config<2> conf = packed_config_list[packed_config_id];
				int y = conf.get_val(0);
				int output_feature_map_id = conf.get_val(1);
				int input_elem_id = (entry_id * input_feature_map_count * input_height + y) * input_width + x;
				const float * current_weights = weights + (int)(weight_count_per_output_feature_map * output_feature_map_id);

				float bias_list[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					if (i < output_feature_map_count - output_feature_map_id)
						bias_list[i] = biases[output_feature_map_id + i];
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = bias_list[i];
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count; ++input_layer_id)
				{
					for(int input_y = 0; input_y < window_height; ++input_y)
					{
						#pragma unroll 4
						for(int input_x = 0; input_x < window_width; ++input_x)
						{
							float weight_list[FEATURE_MAP_BLOCK_SIZE];
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								weight_list[i] = current_weights[weight_offsets[i]];

							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								float inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
							}
							current_weights++;
							input_elem_id++;
						}
						input_elem_id += input_width - window_width;
					}
					input_elem_id += input_width * (input_height - window_height);
				}

				float * base_output = output + ((entry_id * output_feature_map_count + output_feature_map_id) * output_height + y) * output_width + x;
				int output_neuron_count_per_feature_map = output_height * output_width;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_width - x)
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, int BLOCK_SIZE>
		__global__ void convolution_2d_tex_exact_blocked_kernel_fermi(
			float * __restrict output,
			const float * __restrict weights,
			const float * __restrict biases,
			const packed_config<2> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int input_width,
			int input_height,
			int window_height,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < output_width) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = WINDOW_WIDTH * window_height * input_feature_map_count;
				packed_config<2> conf = packed_config_list[packed_config_id];
				int y = conf.get_val(0);
				int output_feature_map_id = conf.get_val(1);
				int input_elem_id = (entry_id * input_feature_map_count * input_height + y) * input_width + x;
				const float * current_weights = weights + (int)(weight_count_per_output_feature_map * output_feature_map_id);

				float bias_list[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					if (i < output_feature_map_count - output_feature_map_id)
						bias_list[i] = biases[output_feature_map_id + i];
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = bias_list[i];
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count; ++input_layer_id)
				{
					for(int input_y = 0; input_y < window_height; ++input_y)
					{
						#pragma unroll
						for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
						{
							float weight_list[FEATURE_MAP_BLOCK_SIZE];
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								weight_list[i] = current_weights[weight_offsets[i]];

							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								float inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
							}
							current_weights++;
							input_elem_id++;
						}
						input_elem_id += input_width - WINDOW_WIDTH;
					}
					input_elem_id += input_width * (input_height - window_height);
				}

				float * base_output = output + ((entry_id * output_feature_map_count + output_feature_map_id) * output_height + y) * output_width + x;
				int output_neuron_count_per_feature_map = output_height * output_width;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_width - x)
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		convolution_2d_layer_tester_cuda_fermi::convolution_2d_layer_tester_cuda_fermi()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
		}

		convolution_2d_layer_tester_cuda_fermi::~convolution_2d_layer_tester_cuda_fermi()
		{
		}

#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

#define launch_exact_kernel_const_const(window_width_const, block_size_const) \
	convolution_2d_tex_exact_blocked_kernel_fermi<window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*additional_buffers[0], *data[0], *data[1], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], window_sizes[1], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count, packed_config_count);

#define launch_exact_kernel_const(window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const(1, block_size_const); \
			break; \
		case 2: \
			launch_exact_kernel_const_const(2, block_size_const); \
			break; \
		case 3: \
			launch_exact_kernel_const_const(3, block_size_const); \
			break; \
		case 4: \
			launch_exact_kernel_const_const(4, block_size_const); \
			break; \
		case 5: \
			launch_exact_kernel_const_const(5, block_size_const); \
			break; \
		case 6: \
			launch_exact_kernel_const_const(6, block_size_const); \
			break; \
		case 7: \
			launch_exact_kernel_const_const(7, block_size_const); \
			break; \
		case 8: \
			launch_exact_kernel_const_const(8, block_size_const); \
			break; \
		case 9: \
			launch_exact_kernel_const_const(9, block_size_const); \
			break; \
		case 10: \
			launch_exact_kernel_const_const(10, block_size_const); \
			break; \
		};

#define launch_exact_kernel(window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_exact_kernel_const(window_width, 1); \
			break; \
		case 2: \
			launch_exact_kernel_const(window_width, 2); \
			break; \
		case 3: \
			launch_exact_kernel_const(window_width, 3); \
			break; \
		case 4: \
			launch_exact_kernel_const(window_width, 4); \
			break; \
		case 5: \
			launch_exact_kernel_const(window_width, 5); \
			break; \
		};

#define launch_kernel_const(block_size_const) \
	convolution_2d_tex_blocked_kernel_fermi<block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*additional_buffers[0], *data[0], *data[1], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], window_sizes[0], window_sizes[1], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count, packed_config_count);

#define launch_kernel(block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_kernel_const(1); \
			break; \
		case 2: \
			launch_kernel_const(2); \
			break; \
		case 3: \
			launch_kernel_const(3); \
			break; \
		case 4: \
			launch_kernel_const(4); \
			break; \
		case 5: \
			launch_kernel_const(5); \
			break; \
		};

		void convolution_2d_layer_tester_cuda_fermi::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_tex_ref, *input_buffer, desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			int packed_config_count =  output_configuration_specific.dimension_sizes[1] * forward_output_feature_map_block_count;
			const packed_config<2> * packed_config_list = static_cast<const packed_config<2> *>((const void *)*additional_buffers[1]);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				forward_x_block_count,
				packed_config_count,
				entry_count);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				launch_exact_kernel(window_sizes[0], forward_x_block_size);
			}
			else
			{
				launch_kernel(forward_x_block_size);
			}
		}

		int convolution_2d_layer_tester_cuda_fermi::get_block_size(int output_width)
		{
			int block_count = (output_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
			int block_size = (output_width + block_count - 1) / block_count;
			return block_size;
		}

		void convolution_2d_layer_tester_cuda_fermi::tester_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_sizes.push_back(static_cast<int>(*it));

			forward_x_block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
			forward_x_block_count = (output_configuration_specific.dimension_sizes[0] + forward_x_block_size - 1) / forward_x_block_size;
			forward_output_feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;
		}

		std::vector<size_t> convolution_2d_layer_tester_cuda_fermi::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		std::vector<unsigned int> convolution_2d_layer_tester_cuda_fermi::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);

			return res;
		}

		cuda_linear_buffer_device_smart_ptr convolution_2d_layer_tester_cuda_fermi::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		std::vector<size_t> convolution_2d_layer_tester_cuda_fermi::get_sizes_of_additional_buffers_fixed() const
		{
			std::vector<size_t> res;

			res.push_back(sizeof(packed_config<2>) * output_configuration_specific.dimension_sizes[1] * forward_output_feature_map_block_count);

			return res;
		}

		void convolution_2d_layer_tester_cuda_fermi::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
			{
				std::vector<packed_config<2> > task_list;
				for(int output_feature_map_block_id = 0; output_feature_map_block_id < forward_output_feature_map_block_count; ++output_feature_map_block_id)
				{
					for(int y = 0; y < output_configuration_specific.dimension_sizes[1]; ++y)
					{
						packed_config<2> new_elem;
						new_elem.set_val(0, y);
						new_elem.set_val(1, output_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
						task_list.push_back(new_elem);
					}
				}
				/*
				std::vector<std::vector<int> > ordered_list;
				std::vector<int> size_list;
				const int scaling_factor = FEATURE_MAP_BLOCK_SIZE;
				size_list.push_back((output_configuration_specific.dimension_sizes[1] + scaling_factor - 1) / scaling_factor);
				size_list.push_back(forward_output_feature_map_block_count);
				space_filling_curve::get_space_filling_curve()->fill_tiling_pattern(size_list, ordered_list);
				for(std::vector<std::vector<int> >::const_iterator it = ordered_list.begin(); it != ordered_list.end(); ++it)
				{
					packed_config<2> new_elem;
					new_elem.set_val(1, it->at(1) * FEATURE_MAP_BLOCK_SIZE);
					int current_y = it->at(0) * scaling_factor;
					for(int i = 0; (i < scaling_factor) && (current_y < output_configuration_specific.dimension_sizes[1]); ++i, ++current_y)
					{
						new_elem.set_val(0, current_y);
						task_list.push_back(new_elem);
					}
				}
				*/

				cuda_safe_call(cudaMemcpy(*additional_buffers[1], &(*task_list.begin()), sizeof(packed_config<2>) * task_list.size(), cudaMemcpyHostToDevice));
			}
		}
	}
}
