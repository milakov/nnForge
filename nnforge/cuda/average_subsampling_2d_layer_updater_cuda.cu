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

#include "average_subsampling_2d_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../average_subsampling_layer.h"
#include "../nn_types.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;

__global__ void average_subsampling_2d_tex_upd_kernel(
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
__global__ void average_subsampling_2d_tex_exact_upd_kernel(
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

extern __shared__ float arr[];

__global__ void average_subsampling_2d_deriviative_upd_kernel(
	float * __restrict input_errors,
	const float * __restrict output_errors,
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
	int threadblock_size = blockDim.x * blockDim.y * blockDim.z;
	int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;

	int * offsets = (int *)arr;
	float * vals = (float *)(arr + threadblock_size * subsampling_width);

	bool in_bounds = (output_x < output_width) && (output_y < output_height) && (feature_map_id < feature_map_count) && (entry_id < entry_count);
	int input_x = output_x * subsampling_width;
	int input_y = output_y * subsampling_height;

	int current_input_errors_elem_id;
	float error;
	if (in_bounds)
	{
		error = output_errors[((entry_id * feature_map_count + feature_map_id) * output_height + output_y) * output_width + output_x] * subsampling_weight;
		current_input_errors_elem_id = ((entry_id * feature_map_count + feature_map_id) * input_height + input_y) * input_width + input_x;
	}

	for(int j = 0; j < subsampling_height; ++j)
	{
		int local_id = thread_id * subsampling_width;
		for(int i = 0; i < subsampling_width; ++i)
		{
			offsets[local_id] = in_bounds ? current_input_errors_elem_id : -1;
			if (in_bounds)
				vals[local_id] = error;
			current_input_errors_elem_id++;
			local_id++;
		}
		__syncthreads();
		local_id = thread_id;
		for(int i = 0; i < subsampling_width; ++i)
		{
			int offset = offsets[local_id];
			float val = vals[local_id];
			if (offset >= 0)
				input_errors[offset] = val;
			local_id += threadblock_size;
		}
		current_input_errors_elem_id += (input_width - subsampling_width);
		__syncthreads();
	}
}

template<int SUBSAMPLING_WIDTH, int SUBSAMPLING_HEIGHT>
__global__ void average_subsampling_2d_deriviative_exact_upd_kernel(
	float * __restrict input_errors,
	const float * __restrict output_errors,
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
	int threadblock_size = blockDim.x * blockDim.y * blockDim.z;
	int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;

	int * offsets = (int *)arr;
	float * vals = (float *)(arr + threadblock_size * SUBSAMPLING_WIDTH);

	bool in_bounds = (output_x < output_width) && (output_y < output_height) && (feature_map_id < feature_map_count) && (entry_id < entry_count);
	int input_x = output_x * SUBSAMPLING_WIDTH;
	int input_y = output_y * SUBSAMPLING_HEIGHT;

	int current_input_errors_elem_id;
	float error;
	if (in_bounds)
	{
		error = output_errors[((entry_id * feature_map_count + feature_map_id) * output_height + output_y) * output_width + output_x] * (1.0F / (float)(SUBSAMPLING_WIDTH * SUBSAMPLING_HEIGHT));
		current_input_errors_elem_id = ((entry_id * feature_map_count + feature_map_id) * input_height + input_y) * input_width + input_x;
	}

	#pragma unroll
	for(int j = 0; j < SUBSAMPLING_HEIGHT; ++j)
	{
		int local_id = thread_id * SUBSAMPLING_WIDTH;
		#pragma unroll
		for(int i = 0; i < SUBSAMPLING_WIDTH; ++i)
		{
			offsets[local_id] = in_bounds ? current_input_errors_elem_id : -1;
			if (in_bounds)
				vals[local_id] = error;
			current_input_errors_elem_id++;
			local_id++;
		}
		__syncthreads();
		local_id = thread_id;
		#pragma unroll
		for(int i = 0; i < SUBSAMPLING_WIDTH; ++i)
		{
			int offset = offsets[local_id];
			float val = vals[local_id];
			if (offset >= 0)
				input_errors[offset] = val;
			local_id += threadblock_size;
		}
		current_input_errors_elem_id += (input_width - SUBSAMPLING_WIDTH);

		if (j < (SUBSAMPLING_HEIGHT - 1))
			__syncthreads();
	}
}

namespace nnforge
{
	namespace cuda
	{
		average_subsampling_2d_layer_updater_cuda::average_subsampling_2d_layer_updater_cuda()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
		}

		average_subsampling_2d_layer_updater_cuda::~average_subsampling_2d_layer_updater_cuda()
		{
		}

#define MAX_WINDOW_WIDTH 4
#define MAX_WINDOW_HEIGHT 4

#define launch_exact_kernel_const_const(window_width_const, window_height_const) \
	average_subsampling_2d_tex_exact_upd_kernel<window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer,input_configuration_specific.dimension_sizes[0],input_configuration_specific.dimension_sizes[1],output_configuration_specific.dimension_sizes[0],output_configuration_specific.dimension_sizes[1],output_configuration_specific.feature_map_count,entry_count);

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

#define launch_backprop_exact_kernel_const_const(window_width_const, window_height_const) \
	average_subsampling_2d_deriviative_exact_upd_kernel<window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(*input_errors_buffer,*output_errors_buffer,input_configuration_specific.dimension_sizes[0],input_configuration_specific.dimension_sizes[1],output_configuration_specific.dimension_sizes[0],output_configuration_specific.dimension_sizes[1],output_configuration_specific.feature_map_count,entry_count);

#define launch_backprop_exact_kernel_const(window_width, window_height_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const_const(1, window_height_const); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const_const(2, window_height_const); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const_const(3, window_height_const); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const_const(4, window_height_const); \
			break; \
		};

#define launch_backprop_exact_kernel(window_width, window_height) \
	switch (window_height) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const(window_width, 1); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const(window_width, 2); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const(window_width, 3); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const(window_width, 4); \
			break; \
		};

		void average_subsampling_2d_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("average_subsampling_2d_layer_updater_cuda is not able to run using offset");

			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_tex_ref, *input_neurons_buffer, desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			int output_elem_count_per_feature_map_aligned = cuda_util::get_power2_aligned_size(output_configuration_specific.dimension_sizes[0]) * output_configuration_specific.dimension_sizes[1];
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map_aligned,
				output_configuration_specific.feature_map_count,
				entry_count);

			if ((subsampling_sizes[0] <= MAX_WINDOW_WIDTH) && (subsampling_sizes[1] <= MAX_WINDOW_HEIGHT))
			{
				launch_exact_kernel(subsampling_sizes[0], subsampling_sizes[1]);
			}
			else
			{
				average_subsampling_2d_tex_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_neurons_buffer,
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

		void average_subsampling_2d_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			if (!is_even_subsampling)
				cuda_util::set_with_value(*cuda_config, *input_errors_buffer, 0.0F, input_configuration_specific.get_neuron_count() * entry_count, stream_id);

			int output_elem_count_per_feature_map_aligned = cuda_util::get_power2_aligned_size(output_configuration_specific.dimension_sizes[0]) * output_configuration_specific.dimension_sizes[1];
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map_aligned,
				output_configuration_specific.feature_map_count,
				entry_count);
			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = threadblock_size * subsampling_sizes[0] * 2 * sizeof(float);

			if ((subsampling_sizes[0] <= MAX_WINDOW_WIDTH) && (subsampling_sizes[1] <= MAX_WINDOW_HEIGHT))
			{
				launch_backprop_exact_kernel(subsampling_sizes[0], subsampling_sizes[1]);
			}
			else
			{
				average_subsampling_2d_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
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

		std::vector<unsigned int> average_subsampling_2d_layer_updater_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);

			return res;
		}

		void average_subsampling_2d_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);

			subsampling_sizes = layer_derived->subsampling_sizes;
			subsampling_weight = 1.0F / static_cast<float>(subsampling_sizes[0] * subsampling_sizes[1]);

			is_even_subsampling = true;
			for(int i = 0; i < subsampling_sizes.size(); ++i)
				if (subsampling_sizes[i] * output_configuration_specific.dimension_sizes[i] != input_configuration_specific.dimension_sizes[i])
				{
					is_even_subsampling = false;
					break;
				}
		}

		bool average_subsampling_2d_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}
	}
}
