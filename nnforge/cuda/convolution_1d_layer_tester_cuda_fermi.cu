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

#include "convolution_1d_layer_tester_cuda_fermi.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "cuda_util.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;

#define FEATURE_MAP_BLOCK_SIZE 4

template<int BLOCK_SIZE>
__global__ void convolution_1d_tex_blocked_kernel_fermi(
	float * __restrict output,
	const float * __restrict weights,
	const float * __restrict biases,
	int output_width,
	int input_width,
	int window_width,
	int input_feature_map_count,
	int output_feature_map_count,
	int entry_count)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
	int output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (x < output_width) && (output_feature_map_id < output_feature_map_count);
	if (in_bounds)
	{
		int weight_count_per_output_feature_map = window_width * input_feature_map_count;
		int input_elem_id = entry_id * input_feature_map_count * input_width + x;
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

		float * base_output = output + (entry_id * output_feature_map_count + output_feature_map_id) * output_width + x;
		#pragma unroll
		for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
		{
			if (i < output_feature_map_count - output_feature_map_id)
			{
				#pragma unroll
				for(int j = 0; j < BLOCK_SIZE; ++j)
				{
					if (j < output_width - x)
						base_output[j + output_width * i] = sums[i * BLOCK_SIZE + j];
				}
			}
		}
	}
}

template<int WINDOW_WIDTH, int BLOCK_SIZE>
__global__ void convolution_1d_tex_exact_blocked_kernel_fermi(
	float * __restrict output,
	const float * __restrict weights,
	const float * __restrict biases,
	int output_width,
	int input_width,
	int input_feature_map_count,
	int output_feature_map_count,
	int entry_count)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
	int output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (x < output_width) && (output_feature_map_id < output_feature_map_count);
	if (in_bounds)
	{
		int weight_count_per_output_feature_map = WINDOW_WIDTH * input_feature_map_count;
		int input_elem_id = entry_id * input_feature_map_count * input_width + x;
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

		float * base_output = output + (entry_id * output_feature_map_count + output_feature_map_id) * output_width + x;
		#pragma unroll
		for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
		{
			if (i < output_feature_map_count - output_feature_map_id)
			{
				#pragma unroll
				for(int j = 0; j < BLOCK_SIZE; ++j)
				{
					if (j < output_width - x)
						base_output[j + output_width * i] = sums[i * BLOCK_SIZE + j];
				}
			}
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		convolution_1d_layer_tester_cuda_fermi::convolution_1d_layer_tester_cuda_fermi()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
		}

		convolution_1d_layer_tester_cuda_fermi::~convolution_1d_layer_tester_cuda_fermi()
		{
		}

#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

#define launch_exact_kernel_const_const(window_width_const, block_size_const) \
	convolution_1d_tex_exact_blocked_kernel_fermi<window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*additional_buffers[0], *data[0], *data[1], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

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
	convolution_1d_tex_blocked_kernel_fermi<block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*additional_buffers[0], *data[0], *data[1], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], window_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

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

		void convolution_1d_layer_tester_cuda_fermi::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_tex_ref, *input_buffer, desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			int block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				(output_configuration_specific.dimension_sizes[0] + block_size - 1) / block_size,
				((output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE),
				entry_count);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				launch_exact_kernel(window_sizes[0], block_size);
			}
			else
			{
				launch_kernel(block_size);
			}
		}

		int convolution_1d_layer_tester_cuda_fermi::get_block_size(int output_width)
		{
			int block_count = (output_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
			int block_size = (output_width + block_count - 1) / block_count;
			return block_size;
		}

		void convolution_1d_layer_tester_cuda_fermi::tester_configured()
		{
			std::tr1::shared_ptr<const convolution_layer> layer_derived = std::tr1::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_sizes.push_back(static_cast<int>(*it));
		}

		std::vector<size_t> convolution_1d_layer_tester_cuda_fermi::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		std::vector<unsigned int> convolution_1d_layer_tester_cuda_fermi::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);

			return res;
		}

		cuda_linear_buffer_device_smart_ptr convolution_1d_layer_tester_cuda_fermi::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}
	}
}
