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

#include "fully_connected_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

extern __shared__ float arr_sh[];

template<bool different_input>
__global__ void fully_connected_upd_kernel(
	float * output,
	const float * __restrict input,
	const float * weights,
	int input_neuron_count,
	int output_neuron_count,
	int min_iteration_count)
{
	int thread_id = threadIdx.x;
	int output_neuron_id = blockIdx.y;
	int entry_id = blockIdx.z;
	int threadblock_size = blockDim.x;

	float sum = 0.0F;
	const float * current_input = input + (int)(different_input ? (entry_id * input_neuron_count): 0);
	const float * current_weights = weights + (int)((entry_id * output_neuron_count + output_neuron_id) * input_neuron_count);
	int current_input_neuron_id = thread_id;
	for(int i = 0; i < min_iteration_count; ++i)
	{
		sum += current_input[current_input_neuron_id] * current_weights[current_input_neuron_id];
		current_input_neuron_id += threadblock_size;
	}
	if (current_input_neuron_id < input_neuron_count)
		sum += current_input[current_input_neuron_id] * current_weights[current_input_neuron_id];

	int lane_id = thread_id & 31;

#if __CUDA_ARCH__ >= 300
	#pragma unroll
	for(int tx = 16; tx > 0; tx >>= 1)
	{
		sum += __shfl_down(sum, tx);
	}
#else
	volatile float * arr = arr_sh;
	arr[thread_id] = sum;
	#pragma unroll
	for(int tx = 16; tx > 0; tx >>= 1)
	{
		if (lane_id < tx)
			arr[thread_id] += arr[thread_id + tx];
	}
	sum = arr[thread_id];
#endif

	if (lane_id == 0)
		atomicAdd(output + entry_id * output_neuron_count + output_neuron_id, sum);
}

template<bool single_output_group_count>
__global__ void fully_connected_deriviative_upd_kernel(
	float * __restrict input_errors,
	const float * __restrict output_errors,
	const float * __restrict weights,
	int input_neuron_count,
	int output_neuron_count,
	int output_group_count,
	int max_iteration_count,
	int entry_count)
{
	int input_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_group_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	bool in_bounds = (input_neuron_id < input_neuron_count) && (output_group_id < output_group_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int output_offset = entry_id * output_neuron_count + output_group_id;
		int weights_offset = (entry_id * output_neuron_count + output_group_id) * input_neuron_count + input_neuron_id;
		int iteration_count = ((max_iteration_count - 1) * output_group_count + output_group_id < output_neuron_count) ? max_iteration_count : max_iteration_count - 1;
		float sum = 0.0F;
		#pragma unroll 4
		for(int i = 0; i < iteration_count; ++i)
		{
			sum += output_errors[output_offset] * weights[weights_offset];
			weights_offset += input_neuron_count * output_group_count;
			output_offset += output_group_count;
		}

		float * current_input = input_errors + entry_id * input_neuron_count + input_neuron_id;
		if (single_output_group_count)
		{
			*current_input = sum;
		}
		else
		{
			atomicAdd(current_input, sum);
		}
	}
}

__global__ void fully_connected_update_biases_upd_kernel(
	float * __restrict biases,
	const float * __restrict output_errors,
	const float * __restrict learning_rate,
	int output_neuron_count,
	int entry_count)
{
	int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	bool in_bounds = (output_neuron_id < output_neuron_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int offset = entry_id * output_neuron_count + output_neuron_id;
		float upd_val = output_errors[offset] * learning_rate[offset] + biases[offset];
		biases[offset] = upd_val;
	}
}

template<bool different_input>
__global__ void fully_connected_update_weights_upd_kernel(
	float * __restrict weights,
	const float * __restrict input_neurons,
	const float * __restrict output_errors,
	const float * __restrict learning_rate,
	int input_neuron_count,
	int output_neuron_count,
	int entry_count,
	float weight_decay)
{
	int input_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_neuron_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	bool in_bounds = (input_neuron_id < input_neuron_count) && (output_neuron_id < output_neuron_count) && (entry_id < entry_count);
	if (in_bounds)
	{
		int input_offset = (different_input ? entry_id * input_neuron_count : 0) + input_neuron_id;
		int offset = (entry_id * output_neuron_count + output_neuron_id) * input_neuron_count + input_neuron_id;
		float current_weight = weights[offset];
		float grd = input_neurons[input_offset] * output_errors[entry_id * output_neuron_count + output_neuron_id];
		float lr = learning_rate[offset];
		float new_weight = current_weight + lr * (grd - weight_decay * current_weight);
		weights[offset] = new_weight;
	}
}

namespace nnforge
{
	namespace cuda
	{
		fully_connected_layer_updater_cuda::fully_connected_layer_updater_cuda()
		{
		}

		fully_connected_layer_updater_cuda::~fully_connected_layer_updater_cuda()
		{
		}

		void fully_connected_layer_updater_cuda::enqueue_test(
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
			cuda_util::copy_buffer(
				*cuda_config,
				*data[1],
				*output_neurons_buffer,
				output_elem_count_per_entry * entry_count,
				stream_id);

			int threadblock_size = get_threadblock_size_forward(input_elem_count_per_entry);
			dim3 grid_size(1, output_elem_count_per_entry, entry_count);
			dim3 block_size(threadblock_size, 1, 1);
			int smem_size = (cuda_config->get_compute_capability() >= 300) ? 0 : (threadblock_size * sizeof(float));
			int min_iteration_count = input_elem_count_per_entry / threadblock_size;

			if (different_input)
			{
				fully_connected_upd_kernel<true><<<grid_size, block_size, smem_size, stream_id>>>(
					*output_neurons_buffer,
					*input_neurons_buffer,
					*data[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					min_iteration_count);
			}
			else
			{
				fully_connected_upd_kernel<false><<<grid_size, block_size, smem_size, stream_id>>>(
					*output_neurons_buffer,
					(const float *)(*input_neurons_buffer) + (offset_input_entry_id * input_elem_count_per_entry),
					*data[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					min_iteration_count);
			}
		}

		void fully_connected_layer_updater_cuda::enqueue_backprop(
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
			if (!different_input)
				throw neural_network_exception("fully_connected_layer_updater_cuda is not able to backprop to the same input");

			int output_group_count = cuda_util::get_group_count(
					*cuda_config,
					input_elem_count_per_entry * entry_count,
					output_elem_count_per_entry);
			int max_iteration_count = (output_elem_count_per_entry + output_group_count - 1) / output_group_count;

			if (output_group_count > 1)
				cuda_util::set_with_value(
					*cuda_config,
					*input_errors_buffer,
					0.0F,
					input_elem_count_per_entry * entry_count,
					stream_id);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_entry,
				output_group_count,
				entry_count);

			if (output_group_count == 1)
				fully_connected_deriviative_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
					*data[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					output_group_count,
					max_iteration_count,
					entry_count);
			else
				fully_connected_deriviative_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
					*data[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					output_group_count,
					max_iteration_count,
					entry_count);
		}

		void fully_connected_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& learning_rate,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			float weight_decay)
		{
			// Update biases
			{
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					output_elem_count_per_entry,
					entry_count,
					1);
				fully_connected_update_biases_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*data[1],
					*output_errors_buffer,
					*learning_rate[1],
					output_elem_count_per_entry,
					entry_count);
			}

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access_x_aligned(
				*cuda_config,
				input_elem_count_per_entry,
				output_elem_count_per_entry,
				entry_count);
			if (different_input)
			{
				fully_connected_update_weights_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*data[0],
					*input_neurons_buffer,
					*output_errors_buffer,
					*learning_rate[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					entry_count,
					weight_decay);
			}
			else
			{
				fully_connected_update_weights_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*data[0],
					(const float *)(*input_neurons_buffer) + (offset_input_entry_id * input_elem_count_per_entry),
					*output_errors_buffer,
					*learning_rate[0],
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					entry_count,
					weight_decay);
			}
		}

		bool fully_connected_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		int fully_connected_layer_updater_cuda::get_threadblock_size_forward(int input_neuron_count)
		{
			int threadblock_size;

			if (input_neuron_count < 128)
			{
				threadblock_size = (input_neuron_count + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (input_neuron_count + 128 - 1) / 128;
				threadblock_size = (input_neuron_count + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}

		std::vector<unsigned int> fully_connected_layer_updater_cuda::get_incoming_weight_count_per_output_neuron_list() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);
			res.push_back(1);

			return res;
		}
	}
}
