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

#include "convolution_2d_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "cuda_util.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;

template<bool different_input>
__global__ void convolution_2d_tex_upd_kernel(
	float * __restrict output,
	const float * __restrict weights,
	const float * __restrict biases,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_width,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int texture_offset,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int _y = blockIdx.y * blockDim.y + threadIdx.y;
	int output_feature_map_id = _y / output_height;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (x < output_width) && (output_feature_map_id < output_feature_map_count);
	if (in_bounds)
	{
		int y = _y - (output_feature_map_id * output_height);
		int input_elem_id = ((different_input ? entry_id * input_feature_map_count * input_height : 0) + y) * input_width + x + texture_offset;
		const float * current_weights = weights + (int)((entry_id * output_feature_map_count + output_feature_map_id) * window_width * window_height * input_feature_map_count);

		float sum = biases[output_feature_map_id + entry_id * output_feature_map_count];

		for(int input_layer_id = 0; input_layer_id < input_feature_map_count; ++input_layer_id)
		{
			for(int input_y = 0; input_y < window_height; ++input_y)
			{
				#pragma unroll 4
				for(int input_x = 0; input_x < window_width; ++input_x)
				{
					sum += tex1Dfetch(input_tex_ref, input_elem_id) * *current_weights;
					current_weights++;
					input_elem_id++;
				}
				input_elem_id += input_width - window_width;
			}
			input_elem_id += input_width * (input_height - window_height);
		}

		output[((entry_id * output_feature_map_count + output_feature_map_id) * output_height + y) * output_width + x] = sum;
	}
}

template<int WINDOW_WIDTH, int BLOCK_SIZE, bool different_input>
__global__ void convolution_2d_tex_exact_blocked_upd_kernel(
	float * __restrict output,
	const float * __restrict weights,
	const float * __restrict biases,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int block_count,
	int input_feature_map_group_count,
	int input_feature_map_group_size,
	int texture_offset,
	int entry_count)
{
	int xy = blockIdx.x * blockDim.x + threadIdx.x;
	int y = xy / block_count;
	int dd = blockIdx.y * blockDim.y + threadIdx.y;
	int input_feature_map_group_id = dd / output_feature_map_count;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (y < output_height) && (input_feature_map_group_id < input_feature_map_group_count);
	if (in_bounds)
	{
		int output_feature_map_id = dd - (input_feature_map_group_id * output_feature_map_count);
		int base_input_feature_map_id = input_feature_map_group_id * input_feature_map_group_size;
		int x = (xy - (y * block_count)) * BLOCK_SIZE;
		int input_elem_id = ((((different_input ? entry_id * input_feature_map_count : 0) + base_input_feature_map_id) * input_height) + y) * input_width + x + texture_offset;
		const float * current_weights = weights + (int)(((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + base_input_feature_map_id) * WINDOW_WIDTH * window_height);
		int iteration_count = min(input_feature_map_group_size, input_feature_map_count - base_input_feature_map_id);

		float initial_value = 0.0F;
		if (input_feature_map_group_id == 0)
			initial_value = biases[output_feature_map_id + entry_id * output_feature_map_count];
		float sums[BLOCK_SIZE];
		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE; ++i)
			sums[i] = initial_value;

		for(int i = 0; i < iteration_count; ++i)
		{
			for(int input_y = 0; input_y < window_height; ++input_y)
			{
				#pragma unroll
				for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
				{
					float weight = *current_weights;
					#pragma unroll
					for(int i = 0; i < BLOCK_SIZE; ++i)
						sums[i] += tex1Dfetch(input_tex_ref, input_elem_id + i) * weight;
					current_weights++;
					input_elem_id++;
				}
				input_elem_id += input_width - WINDOW_WIDTH;
			}
			input_elem_id += input_width * (input_height - window_height);
		}

		float * base_output = output + ((entry_id * output_feature_map_count + output_feature_map_id) * output_height + y) * output_width + x;
		if (input_feature_map_group_count == 1)
		{
			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE; ++i)
			{
				if (i < output_width - x)
					base_output[i] = sums[i];
			}
		}
		else
		{
			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE; ++i)
			{
				if (i < output_width - x)
					atomicAdd(base_output + i, sums[i]);
			}
		}
	}
}

extern __shared__ float arr[];
__global__ void convolution_2d_update_biases_upd_kernel(
	float * __restrict biases,
	const float * __restrict output_errors,
	const float * __restrict training_speed,
	int output_feature_map_count,
	int output_elem_count_per_feature_map,
	int min_iteration_count)
{
	int thread_id = threadIdx.x;
	int output_feature_map_id = blockIdx.y;
	int entry_id = blockIdx.z;
	int threadblock_size = blockDim.x;
	float sum = 0.0F;
	const float * current_error = output_errors + (entry_id * output_feature_map_count + output_feature_map_id) * output_elem_count_per_feature_map;
	int current_output_neuron_id = thread_id;
	for(int i = 0; i < min_iteration_count; ++i)
	{
		sum += current_error[current_output_neuron_id];
		current_output_neuron_id += threadblock_size;
	}
	if (current_output_neuron_id < output_elem_count_per_feature_map)
		sum += current_error[current_output_neuron_id];
	arr[thread_id] = sum;
	__syncthreads();

	int offset = entry_id * output_feature_map_count + output_feature_map_id;
	float current_bias_val;
	float current_training_speed_val;
	if (thread_id == 0)
	{
		current_bias_val = biases[offset];
		current_training_speed_val = training_speed[offset];
	}

	int t_add_elems = threadblock_size >> 1;
	int t_working_elems = (threadblock_size + 1) >> 1;
	while (t_add_elems > 0)
	{
		if (thread_id < t_add_elems)
			arr[thread_id] += arr[thread_id + t_working_elems];
		t_add_elems = t_working_elems >> 1;
		t_working_elems = (t_working_elems + 1) >> 1;
		__syncthreads();
	}

	if (thread_id == 0)
		biases[offset] = arr[0] * current_training_speed_val + current_bias_val;
}

texture<float, cudaTextureType1D, cudaReadModeElementType> output_tex_ref;

__global__ void convolution_2d_deriviative_tex_upd_kernel(
	float * __restrict input_errors,
	const float * __restrict weights,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_width,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int entry_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int _y = blockIdx.y * blockDim.y + threadIdx.y;
	int input_feature_map_id = _y / input_height;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (x < input_width) && (input_feature_map_id < input_feature_map_count);
	if (in_bounds)
	{
		int y = _y - (input_feature_map_id * input_height);
		int output_elem_id = (entry_id * output_feature_map_count * output_height + y) * output_width + x;
		const float * current_weights = weights + (int)((entry_id * input_feature_map_count * output_feature_map_count + input_feature_map_id) * window_width * window_height);

		float sum = 0.0F;

		int min_y_exclusive = y - output_height;
		int max_y_inclusive = y;
		int min_x_exclusive = x - output_width;
		int max_x_inclusive = x;
		for(int output_layer_id = 0; output_layer_id < output_feature_map_count; ++output_layer_id)
		{
			for(int input_y = 0; input_y < window_height; ++input_y)
			{
				bool b_fit1 = (input_y > min_y_exclusive) && (input_y <= max_y_inclusive);
				for(int input_x = 0; input_x < window_width; ++input_x)
				{
					bool b_fit2 = b_fit1 && (input_x > min_x_exclusive) && (input_x <= max_x_inclusive);
					if (b_fit2)
						sum += tex1Dfetch(output_tex_ref, output_elem_id) * *current_weights;
					current_weights++;
					output_elem_id--;
				}
				output_elem_id -= output_width - window_width;
			}
			current_weights += window_width * window_height * (input_feature_map_count - 1);
			output_elem_id += output_width * (output_height + window_height);
		}

		input_errors[((entry_id * input_feature_map_count + input_feature_map_id) * input_height + y) * input_width + x] = sum;
	}
}

template<int WINDOW_WIDTH, int BLOCK_SIZE>
__global__ void convolution_2d_deriviative_tex_exact_blocked_upd_kernel(
	float * __restrict input_errors,
	const float * __restrict weights,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int block_count,
	int output_feature_map_group_count,
	int output_feature_map_group_size,
	int entry_count)
{
	int xy = blockIdx.x * blockDim.x + threadIdx.x;
	int y = xy / block_count;
	int dd = blockIdx.y * blockDim.y + threadIdx.y;
	int output_feature_map_group_id = dd / input_feature_map_count;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	bool in_bounds = (entry_id < entry_count) && (y < input_height) && (output_feature_map_group_id < output_feature_map_group_count);
	if (in_bounds)
	{
		int input_feature_map_id = dd - (output_feature_map_group_id * input_feature_map_count);
		int base_output_feature_map_id = output_feature_map_group_id * output_feature_map_group_size;
		int x = (xy - (y * block_count)) * BLOCK_SIZE + (BLOCK_SIZE - 1);
		int output_elem_id = ((entry_id * output_feature_map_count + base_output_feature_map_id) * output_height + y) * output_width + x;
		const float * current_weights = weights + (int)(((entry_id * output_feature_map_count + base_output_feature_map_id) * input_feature_map_count + input_feature_map_id) * WINDOW_WIDTH * window_height);
		int iteration_count = min(output_feature_map_group_size, output_feature_map_count - base_output_feature_map_id);

		float sums[BLOCK_SIZE];
		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE; ++i)
			sums[i] = 0.0F;

		int min_y_exclusive = y - output_height;
		int max_y_inclusive = y;
		int min_x_exclusive = x - output_width;
		int max_x_inclusive = x;

		unsigned int mask = 0;
		for(int i = BLOCK_SIZE + WINDOW_WIDTH - 2; i >= 0; --i)
			mask = mask << 1 | (((i > min_x_exclusive) && (i <= max_x_inclusive)) ? 1 : 0);

		for(int i = 0; i < iteration_count; ++i)
		{
			for(int input_y = 0; input_y < window_height; ++input_y)
			{
				bool b_fit1 = (input_y > min_y_exclusive) && (input_y <= max_y_inclusive);

				float output_vals[BLOCK_SIZE + WINDOW_WIDTH - 1];
				#pragma unroll
				for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH - 1; ++i)
				{
					bool b_fit2 = b_fit1 && (((1 << i) & mask) != 0);
					if (b_fit2)
						output_vals[i] = tex1Dfetch(output_tex_ref, output_elem_id - i);
					else
						output_vals[i] = 0.0F;
				}

				#pragma unroll
				for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
				{
					float weight = *current_weights;
					#pragma unroll
					for(int i = 0; i < BLOCK_SIZE; ++i)
						sums[i] += output_vals[input_x + i] * weight;
					current_weights++;
				}
				output_elem_id -= output_width;
			}
			current_weights += WINDOW_WIDTH * window_height * (input_feature_map_count - 1);
			output_elem_id += output_width * (output_height + window_height);
		}

		float * base_input = input_errors + ((entry_id * input_feature_map_count + input_feature_map_id) * input_height + y) * input_width + x;
		if (output_feature_map_group_count == 1)
		{
			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE; ++i)
			{
				if (i > x - input_width)
					*(base_input - i) = sums[i];
			}
		}
		else
		{
			#pragma unroll
			for(int i = 0; i < BLOCK_SIZE; ++i)
			{
				if (i > x - input_width)
					atomicAdd(base_input - i, sums[i]);
			}
		}
	}
}

template<bool different_input>
__global__ void convolution_2d_update_weights_upd_kernel(
	float * __restrict weights,
	const float * __restrict output_errors,
	const float * __restrict training_speed,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_width,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int texture_offset,
	int entry_count)
{
	int internal_weights_id = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_pair_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	int weight_y = internal_weights_id / window_width;
	int output_feature_map_id = feature_map_pair_id / input_feature_map_count;

	if ((weight_y < window_height) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int weight_x = internal_weights_id - (weight_y * window_width);
		int input_feature_map_id = feature_map_pair_id - (output_feature_map_id * input_feature_map_count);

		const float * current_output_errors = output_errors + ((entry_id * output_feature_map_count + output_feature_map_id) * output_height) * output_width;
		int input_elem_id = (((different_input ? entry_id * input_feature_map_count : 0) + input_feature_map_id) * input_height + weight_y) * input_width + weight_x + texture_offset;

		float sum = 0.0F;
		for(int y = 0; y < output_height; ++y)
		{
			for(int x = 0; x < output_width; ++x)
			{
				float inp = tex1Dfetch(input_tex_ref, input_elem_id);
				sum += *current_output_errors * inp;
				current_output_errors++;
				input_elem_id++;
			}
			input_elem_id += (window_width - 1);
		}

		int offset = (((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + input_feature_map_id) * window_height + weight_y) * window_width + weight_x;
		weights[offset] += sum * training_speed[offset];
	}
}

template<int WINDOW_WIDTH, bool different_input>
__global__ void convolution_2d_update_weights_exact_blocked_upd_kernel(
	float * __restrict weights,
	const float * __restrict output_errors,
	const float * __restrict training_speed,
	int output_width,
	int output_height,
	int input_width,
	int input_height,
	int window_height,
	int input_feature_map_count,
	int output_feature_map_count,
	int output_y_group_count,
	int texture_offset,
	int entry_count)
{
	int dd = blockIdx.x * blockDim.x + threadIdx.x;
	int output_y_group_id = dd / window_height;
	int feature_map_pair_id = blockIdx.y * blockDim.y + threadIdx.y;
	int output_feature_map_id = feature_map_pair_id / input_feature_map_count;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	if ((output_y_group_id < output_y_group_count) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int weight_y = dd - (output_y_group_id * window_height);
		int input_feature_map_id = feature_map_pair_id - (output_feature_map_id * input_feature_map_count);

		const float * current_output_errors = output_errors + ((entry_id * output_feature_map_count + output_feature_map_id) * output_height + output_y_group_id) * output_width;
		int input_elem_id = (((different_input ? entry_id * input_feature_map_count : 0) + input_feature_map_id) * input_height + weight_y + output_y_group_id) * input_width + texture_offset;

		float sums[WINDOW_WIDTH];
		#pragma unroll
		for(int i = 0; i < WINDOW_WIDTH; ++i)
			sums[i] = 0.0F;

		for(int output_y = output_y_group_id; output_y < output_height; output_y += output_y_group_count)
		{
			float input_buf[WINDOW_WIDTH];
			#pragma unroll
			for(int i = 1; i < WINDOW_WIDTH; ++i)
			{
				input_buf[i] = tex1Dfetch(input_tex_ref, input_elem_id);
				++input_elem_id;
			}

			for(int x = 0; x < output_width; ++x)
			{
				float current_output_error = *current_output_errors;

				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH - 1; ++i)
					input_buf[i] = input_buf[i + 1];
				input_buf[WINDOW_WIDTH - 1] = tex1Dfetch(input_tex_ref, input_elem_id);

				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH; ++i)
					sums[i] += current_output_error * input_buf[i];

				current_output_errors++;
				input_elem_id++;
			}

			current_output_errors += output_width * (output_y_group_count - 1);
			input_elem_id += input_width * (output_y_group_count - 1);
		}

		int offset = (((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + input_feature_map_id) * window_height + weight_y) * WINDOW_WIDTH;
		float * cur_weights = weights + offset;
		const float * cur_training_speed = training_speed + offset;
		if (output_y_group_count == 1)
		{
			#pragma unroll
			for(int i = 0; i < WINDOW_WIDTH; ++i)
				cur_weights[i] += sums[i] * cur_training_speed[i];
		}
		else
		{
			#pragma unroll
			for(int i = 0; i < WINDOW_WIDTH; ++i)
				atomicAdd(cur_weights + i, sums[i] * cur_training_speed[i]);
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		convolution_2d_layer_updater_cuda::convolution_2d_layer_updater_cuda()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
			output_tex_ref.addressMode[0] = cudaAddressModeBorder;
			output_tex_ref.normalized = false;
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
		}

		convolution_2d_layer_updater_cuda::~convolution_2d_layer_updater_cuda()
		{
		}

#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

#define launch_exact_block_kernel_const_const(window_width_const, block_size_const, different_input_const) \
	convolution_2d_tex_exact_blocked_upd_kernel<window_width_const,block_size_const,different_input_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, *data[0], *data[1], output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], window_sizes[1], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, block_count, input_feature_map_group_count, input_feature_map_group_size, texture_offset, entry_count);

#define launch_exact_block_kernel_const(window_width, block_size_const, different_input_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_block_kernel_const_const(1, block_size_const, different_input_const); \
			break; \
		case 2: \
			launch_exact_block_kernel_const_const(2, block_size_const, different_input_const); \
			break; \
		case 3: \
			launch_exact_block_kernel_const_const(3, block_size_const, different_input_const); \
			break; \
		case 4: \
			launch_exact_block_kernel_const_const(4, block_size_const, different_input_const); \
			break; \
		case 5: \
			launch_exact_block_kernel_const_const(5, block_size_const, different_input_const); \
			break; \
		case 6: \
			launch_exact_block_kernel_const_const(6, block_size_const, different_input_const); \
			break; \
		case 7: \
			launch_exact_block_kernel_const_const(7, block_size_const, different_input_const); \
			break; \
		case 8: \
			launch_exact_block_kernel_const_const(8, block_size_const, different_input_const); \
			break; \
		case 9: \
			launch_exact_block_kernel_const_const(9, block_size_const, different_input_const); \
			break; \
		case 10: \
			launch_exact_block_kernel_const_const(10, block_size_const, different_input_const); \
			break; \
		};

#define launch_exact_block_kernel(window_width, block_size, different_input_const) \
	switch (block_size) \
		{ \
		case 1: \
			launch_exact_block_kernel_const(window_width, 1, different_input_const); \
			break; \
		case 2: \
			launch_exact_block_kernel_const(window_width, 2, different_input_const); \
			break; \
		case 3: \
			launch_exact_block_kernel_const(window_width, 3, different_input_const); \
			break; \
		case 4: \
			launch_exact_block_kernel_const(window_width, 4, different_input_const); \
			break; \
		case 5: \
			launch_exact_block_kernel_const(window_width, 5, different_input_const); \
			break; \
		};

#define launch_backprop_exact_block_kernel_const_const(window_width_const, block_size_const) \
	convolution_2d_deriviative_tex_exact_blocked_upd_kernel<window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *data[0], output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], window_sizes[1], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, block_count, output_feature_map_group_count, output_feature_map_group_size, entry_count);

#define launch_backprop_exact_block_kernel_const(window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_block_kernel_const_const(1, block_size_const); \
			break; \
		case 2: \
			launch_backprop_exact_block_kernel_const_const(2, block_size_const); \
			break; \
		case 3: \
			launch_backprop_exact_block_kernel_const_const(3, block_size_const); \
			break; \
		case 4: \
			launch_backprop_exact_block_kernel_const_const(4, block_size_const); \
			break; \
		case 5: \
			launch_backprop_exact_block_kernel_const_const(5, block_size_const); \
			break; \
		case 6: \
			launch_backprop_exact_block_kernel_const_const(6, block_size_const); \
			break; \
		case 7: \
			launch_backprop_exact_block_kernel_const_const(7, block_size_const); \
			break; \
		case 8: \
			launch_backprop_exact_block_kernel_const_const(8, block_size_const); \
			break; \
		case 9: \
			launch_backprop_exact_block_kernel_const_const(9, block_size_const); \
			break; \
		case 10: \
			launch_backprop_exact_block_kernel_const_const(10, block_size_const); \
			break; \
		};

#define launch_backprop_exact_block_kernel(window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_exact_block_kernel_const(window_width, 1); \
			break; \
		case 2: \
			launch_backprop_exact_block_kernel_const(window_width, 2); \
			break; \
		case 3: \
			launch_backprop_exact_block_kernel_const(window_width, 3); \
			break; \
		case 4: \
			launch_backprop_exact_block_kernel_const(window_width, 4); \
			break; \
		case 5: \
			launch_backprop_exact_block_kernel_const(window_width, 5); \
			break; \
		};

#define launch_update_weights_exact_block_kernel_const(window_width_const, different_input_const) \
	convolution_2d_update_weights_exact_blocked_upd_kernel<window_width_const, different_input_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*data[0], *output_errors_buffer, *training_speed[0], output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], window_sizes[1], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, output_y_group_count, texture_offset, entry_count);

#define launch_update_weights_exact_block_kernel(window_width, different_input_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_update_weights_exact_block_kernel_const(1, different_input_const); \
			break; \
		case 2: \
			launch_update_weights_exact_block_kernel_const(2, different_input_const); \
			break; \
		case 3: \
			launch_update_weights_exact_block_kernel_const(3, different_input_const); \
			break; \
		case 4: \
			launch_update_weights_exact_block_kernel_const(4, different_input_const); \
			break; \
		case 5: \
			launch_update_weights_exact_block_kernel_const(5, different_input_const); \
			break; \
		case 6: \
			launch_update_weights_exact_block_kernel_const(6, different_input_const); \
			break; \
		case 7: \
			launch_update_weights_exact_block_kernel_const(7, different_input_const); \
			break; \
		case 8: \
			launch_update_weights_exact_block_kernel_const(8, different_input_const); \
			break; \
		case 9: \
			launch_update_weights_exact_block_kernel_const(9, different_input_const); \
			break; \
		case 10: \
			launch_update_weights_exact_block_kernel_const(10, different_input_const); \
			break; \
		};

		void convolution_2d_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			size_t texture_offset;
			cuda_safe_call(cudaBindTexture(&texture_offset, input_tex_ref, (const float *)(*input_neurons_buffer) + (offset_input_entry_id * input_elem_count_per_entry), desc, input_elem_count_per_entry * sizeof(float) * (different_input ? entry_count : 1)));
			texture_offset /= sizeof(float);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				int block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
				int block_count = (output_configuration_specific.dimension_sizes[0] + block_size - 1) / block_size;
				int input_feature_map_group_count = cuda_util::get_group_count(
					*cuda_config,
					block_count * output_configuration_specific.dimension_sizes[1] * output_configuration_specific.feature_map_count * entry_count,
					input_configuration_specific.feature_map_count);
				int input_feature_map_group_size = (input_configuration_specific.feature_map_count + input_feature_map_group_count - 1) / input_feature_map_group_count;

				if (input_feature_map_group_count > 1)
					cuda_util::set_with_value(
						*cuda_config,
						*output_neurons_buffer,
						0.0F,
						output_elem_count_per_entry * entry_count,
						stream_id);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count * output_configuration_specific.dimension_sizes[1],
					output_configuration_specific.feature_map_count * input_feature_map_group_count,
					entry_count);

				if (different_input)
				{
					launch_exact_block_kernel(window_sizes[0], block_size, true);
				}
				else
				{
					launch_exact_block_kernel(window_sizes[0], block_size, false);
				}
			}
			else
			{
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
					*cuda_config,
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1] * output_configuration_specific.feature_map_count,
					entry_count);

				if (different_input)
					convolution_2d_tex_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*output_neurons_buffer,
						*data[0],
						*data[1],
						output_configuration_specific.dimension_sizes[0],
						output_configuration_specific.dimension_sizes[1],
						input_configuration_specific.dimension_sizes[0],
						input_configuration_specific.dimension_sizes[1],
						window_sizes[0],
						window_sizes[1],
						input_configuration_specific.feature_map_count,
						output_configuration_specific.feature_map_count,
						texture_offset,
						entry_count);
				else
					convolution_2d_tex_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*output_neurons_buffer,
						*data[0],
						*data[1],
						output_configuration_specific.dimension_sizes[0],
						output_configuration_specific.dimension_sizes[1],
						input_configuration_specific.dimension_sizes[0],
						input_configuration_specific.dimension_sizes[1],
						window_sizes[0],
						window_sizes[1],
						input_configuration_specific.feature_map_count,
						output_configuration_specific.feature_map_count,
						texture_offset,
						entry_count);
			}
		}

		void convolution_2d_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			if (!different_input)
				throw neural_network_exception("convolution_2d_layer_updater_cuda is not able to backprop to the same input");

			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, output_tex_ref, *output_errors_buffer, desc, output_elem_count_per_entry * entry_count * sizeof(float)));

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				int block_size = get_block_size(input_configuration_specific.dimension_sizes[0]);
				int block_count = (input_configuration_specific.dimension_sizes[0] + block_size - 1) / block_size;
				int output_feature_map_group_count = cuda_util::get_group_count(
					*cuda_config,
					block_count * input_configuration_specific.dimension_sizes[1] * input_configuration_specific.feature_map_count * entry_count,
					output_configuration_specific.feature_map_count);
				int output_feature_map_group_size = (output_configuration_specific.feature_map_count + output_feature_map_group_count - 1) / output_feature_map_group_count;

				if (output_feature_map_group_count > 1)
					cuda_util::set_with_value(
						*cuda_config,
						*input_errors_buffer,
						0.0F,
						input_elem_count_per_entry * entry_count,
						stream_id);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count * input_configuration_specific.dimension_sizes[1],
					input_configuration_specific.feature_map_count * output_feature_map_group_count,
					entry_count);
				launch_backprop_exact_block_kernel(window_sizes[0], block_size);
			}
			else
			{
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
					*cuda_config,
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1] * input_configuration_specific.feature_map_count,
					entry_count);

				convolution_2d_deriviative_tex_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*data[0],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					input_configuration_specific.dimension_sizes[0],
					input_configuration_specific.dimension_sizes[1],
					window_sizes[0],
					window_sizes[1],
					input_configuration_specific.feature_map_count,
					output_configuration_specific.feature_map_count,
					entry_count);
			}
		}

		void convolution_2d_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& training_speed,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			// Update biases
			{
				int threadblock_size = get_threadblock_size_biases(output_elem_count_per_feature_map);
				dim3 grid_size(1, output_configuration_specific.feature_map_count, entry_count);
				dim3 block_size(threadblock_size, 1, 1);
				int smem_size = threadblock_size * sizeof(float);
				int min_iteration_count = output_elem_count_per_feature_map / threadblock_size;

				convolution_2d_update_biases_upd_kernel<<<grid_size, block_size, smem_size, stream_id>>>(
					*data[1],
					*output_errors_buffer,
					*training_speed[1],
					output_configuration_specific.feature_map_count,
					output_elem_count_per_feature_map,
					min_iteration_count);
			}

			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			size_t texture_offset;
			cuda_safe_call(cudaBindTexture(&texture_offset, input_tex_ref, (const float *)(*input_neurons_buffer) + (offset_input_entry_id * input_elem_count_per_entry), desc, input_elem_count_per_entry * sizeof(float) * (different_input ? entry_count : 1)));
			texture_offset /= sizeof(float);

			// Update weights
			{
				if (window_sizes[0] <= MAX_WINDOW_WIDTH)
				{
					int output_y_group_count = cuda_util::get_group_count(
						*cuda_config,
						output_configuration_specific.feature_map_count * input_configuration_specific.feature_map_count * window_sizes[1] * entry_count,
						output_configuration_specific.dimension_sizes[1]);
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						window_sizes[1] * output_y_group_count,
						output_configuration_specific.feature_map_count * input_configuration_specific.feature_map_count,
						entry_count);

					if (different_input)
					{
						launch_update_weights_exact_block_kernel(window_sizes[0], true);
					}
					else
					{
						launch_update_weights_exact_block_kernel(window_sizes[0], false);
					}
				}
				else
				{
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						window_sizes[0] * window_sizes[1],
						output_configuration_specific.feature_map_count * input_configuration_specific.feature_map_count,
						entry_count);

					if (different_input)
						convolution_2d_update_weights_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
							*data[0],
							*output_errors_buffer,
							*training_speed[0],
							output_configuration_specific.dimension_sizes[0],
							output_configuration_specific.dimension_sizes[1],
							input_configuration_specific.dimension_sizes[0],
							input_configuration_specific.dimension_sizes[1],
							window_sizes[0],
							window_sizes[1],
							input_configuration_specific.feature_map_count,
							output_configuration_specific.feature_map_count,
							texture_offset,
							entry_count);
					else
						convolution_2d_update_weights_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
							*data[0],
							*output_errors_buffer,
							*training_speed[0],
							output_configuration_specific.dimension_sizes[0],
							output_configuration_specific.dimension_sizes[1],
							input_configuration_specific.dimension_sizes[0],
							input_configuration_specific.dimension_sizes[1],
							window_sizes[0],
							window_sizes[1],
							input_configuration_specific.feature_map_count,
							output_configuration_specific.feature_map_count,
							texture_offset,
							entry_count);
				}
			}
		}

		int convolution_2d_layer_updater_cuda::get_block_size(int width)
		{
			int block_count = (width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
			int block_size = (width + block_count - 1) / block_count;
			return block_size;
		}

		void convolution_2d_layer_updater_cuda::updater_configured()
		{
			std::tr1::shared_ptr<const convolution_layer> layer_derived = std::tr1::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_sizes.push_back(static_cast<int>(*it));
		}

		bool convolution_2d_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		std::vector<unsigned int> convolution_2d_layer_updater_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);
			res.push_back(output_elem_count_per_entry);

			return res;
		}

		int convolution_2d_layer_updater_cuda::get_threadblock_size_biases(int output_neuron_count)
		{
			if (output_neuron_count < 256)
				return output_neuron_count;

			int threadblock_count = (output_neuron_count + 256 - 1) / 256;
			int threadblock_size = (output_neuron_count + threadblock_count - 1) / threadblock_count;
			threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;

			return threadblock_size;
		}
	}
}
