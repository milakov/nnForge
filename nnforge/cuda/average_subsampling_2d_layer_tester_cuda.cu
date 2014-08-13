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

#include "average_subsampling_2d_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../average_subsampling_layer.h"
#include "../nn_types.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;

__global__ void average_subsampling_2d_tex_kernel(
	float * __restrict output,
	int subsampling_width,
	int subsampling_height,
	float subsampling_weight,
	int input_width,
	int input_height,
	int output_width,
	int output_height,
	int feature_map_count,
	int entry_count)
{
	int elem_id_in_feature_map = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	int tt = 32 - __clz(output_width - 1);
	int output_y = elem_id_in_feature_map >> tt;
	int output_x = elem_id_in_feature_map & ((1 << tt) - 1);

	bool in_bounds = (output_x < output_width) && (output_y < output_height) && (feature_map_id < feature_map_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int input_x = output_x * subsampling_width;
		int input_y = output_y * subsampling_height;

		int current_input_elem_id = ((entry_id * feature_map_count + feature_map_id) * input_height + input_y) * input_width + input_x;

		float sum = 0.0F;
		for(int j = 0; j < subsampling_height; ++j)
		{
			#pragma unroll 4
			for(int i = 0; i < subsampling_width; ++i)
			{
				sum += tex1Dfetch(input_tex_ref, current_input_elem_id);
				current_input_elem_id++;
			}
			current_input_elem_id += (input_width - subsampling_width);
		}

		output[((entry_id * feature_map_count + feature_map_id) * output_height + output_y) * output_width + output_x] = sum * subsampling_weight;
	}
}

template<int SUBSAMPLING_WIDTH, int SUBSAMPLING_HEIGHT>
__global__ void average_subsampling_2d_tex_exact_kernel(
	float * __restrict output,
	int input_width,
	int input_height,
	int output_width,
	int output_height,
	int feature_map_count,
	int entry_count)
{
	int elem_id_in_feature_map = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	int tt = 32 - __clz(output_width - 1);
	int output_y = elem_id_in_feature_map >> tt;
	int output_x = elem_id_in_feature_map & ((1 << tt) - 1);

	bool in_bounds = (output_x < output_width) && (output_y < output_height) && (feature_map_id < feature_map_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int input_x = output_x * SUBSAMPLING_WIDTH;
		int input_y = output_y * SUBSAMPLING_HEIGHT;

		int current_input_elem_id = ((entry_id * feature_map_count + feature_map_id) * input_height + input_y) * input_width + input_x;

		float sum = 0.0F;
		#pragma unroll
		for(int j = 0; j < SUBSAMPLING_HEIGHT; ++j)
		{
			#pragma unroll
			for(int i = 0; i < SUBSAMPLING_WIDTH; ++i)
			{
				sum += tex1Dfetch(input_tex_ref, current_input_elem_id);
				current_input_elem_id++;
			}
			current_input_elem_id += (input_width - SUBSAMPLING_WIDTH);
		}

		output[((entry_id * feature_map_count + feature_map_id) * output_height + output_y) * output_width + output_x] = sum * (1.0F / (float)(SUBSAMPLING_WIDTH * SUBSAMPLING_HEIGHT));
	}
}

namespace nnforge
{
	namespace cuda
	{
		average_subsampling_2d_layer_tester_cuda::average_subsampling_2d_layer_tester_cuda()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
		}

		average_subsampling_2d_layer_tester_cuda::~average_subsampling_2d_layer_tester_cuda()
		{
		}

#define MAX_WINDOW_WIDTH 4
#define MAX_WINDOW_HEIGHT 4

#define launch_exact_kernel_const_const(window_width_const, window_height_const) \
	average_subsampling_2d_tex_exact_kernel<window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(output,input_configuration_specific.dimension_sizes[0],input_configuration_specific.dimension_sizes[1],output_configuration_specific.dimension_sizes[0],output_configuration_specific.dimension_sizes[1],output_configuration_specific.feature_map_count,entry_count);

#define launch_exact_kernel_const(window_width, window_height_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const(1, window_height_const); \
			break; \
		case 2: \
			launch_exact_kernel_const_const(2, window_height_const); \
			break; \
		case 3: \
			launch_exact_kernel_const_const(3, window_height_const); \
			break; \
		case 4: \
			launch_exact_kernel_const_const(4, window_height_const); \
			break; \
		};

#define launch_exact_kernel(window_width, window_height) \
	switch (window_height) \
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
		};

		void average_subsampling_2d_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_tex_ref, *input_buffer, desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			int output_elem_count_per_feature_map_aligned = cuda_util::get_power2_aligned_size(output_configuration_specific.dimension_sizes[0]) * output_configuration_specific.dimension_sizes[1];
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map_aligned,
				output_configuration_specific.feature_map_count,
				entry_count);
			float * output = *additional_buffers[0];

			if ((subsampling_sizes[0] <= MAX_WINDOW_WIDTH) && (subsampling_sizes[1] <= MAX_WINDOW_HEIGHT))
			{
				launch_exact_kernel(subsampling_sizes[0], subsampling_sizes[1]);
			}
			else
			{
				average_subsampling_2d_tex_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					output,
					subsampling_sizes[0],
					subsampling_sizes[1],
					subsampling_weight,
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					output_configuration_specific.feature_map_count,
					entry_count);
			}
		}

		std::vector<size_t> average_subsampling_2d_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		std::vector<unsigned int> average_subsampling_2d_layer_tester_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);

			return res;
		}

		cuda_linear_buffer_device_smart_ptr average_subsampling_2d_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		void average_subsampling_2d_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);

			subsampling_sizes = layer_derived->subsampling_sizes;
			subsampling_weight = 1.0F / static_cast<float>(subsampling_sizes[0] * subsampling_sizes[1]);
		}
	}
}
