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

#include "sigmoid_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "util_cuda.h"

static __forceinline__ __device__ float sigmoid(float x)
{
	return __fdividef(1.0F, 1.0F + __expf(-x));
}

__global__ void sigmoid_upd_kernel(
	const float4 * __restrict input,
	float4 * __restrict output,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = input[elem_id];
		val.x = sigmoid(val.x);
		val.y = sigmoid(val.y);
		val.z = sigmoid(val.z);
		val.w = sigmoid(val.w);
		output[elem_id] = val;
	}
}

static __forceinline__ __device__ float sigmoid_derivative(float x)
{
	return x * (1.0F - x);
}

__global__ void sigmoid_backprop_upd_kernel(
	float4 * __restrict errors,
	const float4 * __restrict output_neurons,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = output_neurons[elem_id];
		float4 current_error = errors[elem_id];
		val.x = sigmoid_derivative(val.x);
		val.y = sigmoid_derivative(val.y);
		val.z = sigmoid_derivative(val.z);
		val.w = sigmoid_derivative(val.w);
		current_error.x *= val.x;
		current_error.y *= val.y;
		current_error.z *= val.z;
		current_error.w *= val.w;
		errors[elem_id] = current_error;
	}
}

namespace nnforge
{
	namespace cuda
	{
		sigmoid_layer_updater_cuda::sigmoid_layer_updater_cuda()
		{
		}

		sigmoid_layer_updater_cuda::~sigmoid_layer_updater_cuda()
		{
		}

		void sigmoid_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("sigmoid_layer_updater_cuda is not able to run using offset");

			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			sigmoid_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				elem_count);
		}

		void sigmoid_layer_updater_cuda::enqueue_backprop(
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
			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			sigmoid_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				elem_count);
		}

		bool sigmoid_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
