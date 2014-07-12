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

#include "local_contrast_subtractive_2d_layer_updater_cuda.h"

#include "../local_contrast_subtractive_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "util_cuda.h"

__global__ void local_contrast_subtractive_2d_blur_horizontal_upd_kernel(
	const float * __restrict input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int window_width,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)(((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll 4
		for(int i = 1; i < window_width; ++i)
		{
			current_weights++;
			if (i < x + 1)
				current_input_low--;
			if (i > x + 1)
				current_input_low++;
			if (i < width - x)
				current_input_high++;
			if (i > width - x)
				current_input_high--;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		output[(z * height + y) * width + x] = res;
	}
}

template<int WINDOW_WIDTH>
__global__ void local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel(
	const float * __restrict input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)(((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll
		for(int i = 1; i < WINDOW_WIDTH; ++i)
		{
			current_weights++;
			if (i < x + 1)
				current_input_low--;
			if (i > x + 1)
				current_input_low++;
			if (i < width - x)
				current_input_high++;
			if (i > width - x)
				current_input_high--;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		output[(z * height + y) * width + x] = res;
	}
}

__global__ void local_contrast_subtractive_2d_blur_vertical_and_subtract_upd_kernel(
	const float * __restrict input,
	const float * __restrict original_input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int window_height,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)((z * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll 4
		for(int i = 1; i < window_height; ++i)
		{
			current_weights++;
			if (i < y + 1)
				current_input_low -= width;
			if (i > y + 1)
				current_input_low += width;
			if (i < height - y)
				current_input_high += width;
			if (i > height - y)
				current_input_high -= width;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		int offset = ((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x;
		output[offset] = original_input[offset] - res;
	}
}

template<int WINDOW_HEIGHT>
__global__ void local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel(
	const float * __restrict input,
	const float * __restrict original_input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)((z * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll
		for(int i = 1; i < WINDOW_HEIGHT; ++i)
		{
			current_weights++;
			if (i < y + 1)
				current_input_low -= width;
			if (i > y + 1)
				current_input_low += width;
			if (i < height - y)
				current_input_high += width;
			if (i > height - y)
				current_input_high -= width;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		int offset = ((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x;
		output[offset] = original_input[offset] - res;
	}
}

__global__ void local_contrast_subtractive_2d_copy_unaffected_upd_kernel(
	const float * __restrict original_input,
	float * __restrict output,
	const unsigned int * __restrict unaffected_feature_map_list,
	int input_feature_map_count,
	int unaffected_feature_map_count,
	int elem_count_per_fature_map,
	int entry_count)
{
	int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
	int unaffected_feature_map_index = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	bool in_bounds = (elem_id < elem_count_per_fature_map) && (unaffected_feature_map_index < unaffected_feature_map_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int unaffected_feature_map_id = unaffected_feature_map_list[unaffected_feature_map_index];
		int offset = (entry_id * input_feature_map_count + unaffected_feature_map_id) * elem_count_per_fature_map + elem_id;
		output[offset] = original_input[offset];
	}
}

__global__ void local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_upd_kernel(
	const float * __restrict input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int window_height,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)((z * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll 4
		for(int i = 1; i < window_height; ++i)
		{
			current_weights++;
			if (i < y + 1)
				current_input_low -= width;
			if (i > y + 1)
				current_input_low += width;
			if (i < height - y)
				current_input_high += width;
			if (i > height - y)
				current_input_high -= width;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		output[((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x] -= res;
	}
}

template<int WINDOW_HEIGHT>
__global__ void local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel(
	const float * __restrict input,
	float * __restrict output,
	const unsigned int * __restrict affected_feature_map_list,
	const float * __restrict weights,
	int input_feature_map_count,
	int affected_feature_map_count,
	int width,
	int height,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int entry_id = z / affected_feature_map_count;

	bool in_bounds = (x < width) && (y < height) && (entry_id < entry_count);
	if (in_bounds)
	{
		int affected_feature_map_index = z - (entry_id * affected_feature_map_count);
		int affected_feature_map_id = affected_feature_map_list[affected_feature_map_index];
		const float * current_input = input + (int)((z * height + y) * width + x);
		const float * current_input_low = current_input;
		const float * current_input_high = current_input;
		const float * current_weights = weights;
		float res = *current_input * *current_weights;
		#pragma unroll
		for(int i = 1; i < WINDOW_HEIGHT; ++i)
		{
			current_weights++;
			if (i < y + 1)
				current_input_low -= width;
			if (i > y + 1)
				current_input_low += width;
			if (i < height - y)
				current_input_high += width;
			if (i > height - y)
				current_input_high -= width;
			res += (*current_input_low + *current_input_high) * *current_weights;
		}

		output[((entry_id * input_feature_map_count + affected_feature_map_id) * height + y) * width + x] -= res;
	}
}

namespace nnforge
{
	namespace cuda
	{
		local_contrast_subtractive_2d_layer_updater_cuda::local_contrast_subtractive_2d_layer_updater_cuda()
		{
		}

		local_contrast_subtractive_2d_layer_updater_cuda::~local_contrast_subtractive_2d_layer_updater_cuda()
		{
		}

		void local_contrast_subtractive_2d_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("local_contrast_subtractive_2d_layer_updater_cuda is not able to run using offset");

			std::pair<dim3, dim3> kernel_1st_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				input_configuration_specific.dimension_sizes[0],
				input_configuration_specific.dimension_sizes[1],
				affected_feature_map_count * entry_count);
			switch(half_window_sizes[0])
			{
			case 1:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<1><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 2:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<2><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 3:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<3><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 4:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<4><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 5:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<5><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 6:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<6><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 7:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<7><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 8:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<8><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 9:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<9><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 10:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<10><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*input_neurons_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			default:
				local_contrast_subtractive_2d_blur_horizontal_upd_kernel<<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(
					*input_neurons_buffer,
					*additional_buffers[0],
					*schema_data[0],
					*schema_data[1],
					input_configuration_specific.feature_map_count,
					affected_feature_map_count,
					half_window_sizes[0],
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					entry_count);
				break;
			}

			std::pair<dim3, dim3> kernel_2nd_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				input_configuration_specific.dimension_sizes[0],
				input_configuration_specific.dimension_sizes[1],
				affected_feature_map_count * entry_count);
			switch(half_window_sizes[1])
			{
			case 1:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<1><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 2:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<2><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 3:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<3><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 4:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<4><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 5:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<5><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 6:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<6><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 7:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<7><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 8:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<8><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 9:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<9><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 10:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_exact_upd_kernel<10><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *input_neurons_buffer, *output_neurons_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			default:
				local_contrast_subtractive_2d_blur_vertical_and_subtract_upd_kernel<<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(
					*additional_buffers[0],
					*input_neurons_buffer,
					*output_neurons_buffer,
					*schema_data[0],
					*schema_data[2],
					input_configuration_specific.feature_map_count,
					affected_feature_map_count,
					half_window_sizes[1],
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					entry_count);
				break;
			}

			if (unaffected_feature_map_count > 0)
			{
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
					*cuda_config,
					input_elem_count_per_feature_map,
					unaffected_feature_map_count,
					entry_count);
				local_contrast_subtractive_2d_copy_unaffected_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_neurons_buffer,
					*output_neurons_buffer,
					*schema_data[3],
					input_configuration_specific.feature_map_count,
					unaffected_feature_map_count,
					input_elem_count_per_feature_map,
					entry_count);
			}
		}

		void local_contrast_subtractive_2d_layer_updater_cuda::enqueue_backprop(
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
			std::pair<dim3, dim3> kernel_1st_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				input_configuration_specific.dimension_sizes[0],
				input_configuration_specific.dimension_sizes[1],
				affected_feature_map_count * entry_count);
			switch(half_window_sizes[0])
			{
			case 1:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<1><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 2:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<2><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 3:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<3><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 4:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<4><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 5:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<5><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 6:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<6><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 7:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<7><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 8:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<8><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 9:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<9><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 10:
				local_contrast_subtractive_2d_blur_horizontal_exact_upd_kernel<10><<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(*output_errors_buffer, *additional_buffers[0], *schema_data[0], *schema_data[1], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			default:
				local_contrast_subtractive_2d_blur_horizontal_upd_kernel<<<kernel_1st_dims.first, kernel_1st_dims.second, 0, stream_id>>>(
					*output_errors_buffer,
					*additional_buffers[0],
					*schema_data[0],
					*schema_data[1],
					input_configuration_specific.feature_map_count,
					affected_feature_map_count,
					half_window_sizes[0],
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					entry_count);
				break;
			}

			std::pair<dim3, dim3> kernel_2nd_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				input_configuration_specific.dimension_sizes[0],
				input_configuration_specific.dimension_sizes[1],
				affected_feature_map_count * entry_count);
			switch(half_window_sizes[1])
			{
			case 1:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<1><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 2:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<2><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 3:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<3><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 4:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<4><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 5:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<5><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 6:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<6><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 7:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<7><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 8:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<8><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 9:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<9><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			case 10:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_exact_upd_kernel<10><<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(*additional_buffers[0], *output_errors_buffer, *schema_data[0], *schema_data[2], input_configuration_specific.feature_map_count, affected_feature_map_count, input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], entry_count);
				break;
			default:
				local_contrast_subtractive_2d_square_deriviative_blur_vertical_and_subtract_upd_kernel<<<kernel_2nd_dims.first, kernel_2nd_dims.second, 0, stream_id>>>(
					*additional_buffers[0],
					*output_errors_buffer,
					*schema_data[0],
					*schema_data[2],
					input_configuration_specific.feature_map_count,
					affected_feature_map_count,
					half_window_sizes[1],
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					entry_count);
				break;
			}
		}

		void local_contrast_subtractive_2d_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const local_contrast_subtractive_layer> layer_derived = nnforge_dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);

			affected_feature_map_count = static_cast<int>(layer_derived->feature_maps_affected.size());
			unaffected_feature_map_count = static_cast<int>(layer_derived->feature_maps_unaffected.size());

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				half_window_sizes.push_back(static_cast<int>((*it + 1) >> 1));

			central_mult = 1.0F - (2.0F * layer_derived->window_weights_list[0][0] * layer_derived->window_weights_list[1][0]);
		}

		std::vector<size_t> local_contrast_subtractive_2d_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_feature_map * affected_feature_map_count * sizeof(float));

			return res;
		}

		bool local_contrast_subtractive_2d_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
