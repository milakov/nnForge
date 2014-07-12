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

#include "soft_rectified_linear_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../neural_network_exception.h"
#include "util_cuda.h"

__global__ void soft_rectified_linear_upd_kernel(
	const float4 * __restrict input,
	float4 * __restrict output,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = input[elem_id];
		val.x = __logf(__expf(val.x) + 1.0F);
		val.y = __logf(__expf(val.y) + 1.0F);
		val.z = __logf(__expf(val.z) + 1.0F);
		val.w = __logf(__expf(val.w) + 1.0F);
		output[elem_id] = val;
	}
}

static __forceinline__ __device__ float soft_rectified_linear_deriviative(float x)
{
	float val2 = __expf(x);
	return __fdividef((val2 - 1.0F), val2);
}

__global__ void soft_rectified_linear_deriviative_upd_kernel(
	float4 * __restrict errors,
	const float4 * __restrict output_neurons,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = output_neurons[elem_id];
		float4 current_error = errors[elem_id];
		val.x = soft_rectified_linear_deriviative(val.x);
		val.y = soft_rectified_linear_deriviative(val.y);
		val.z = soft_rectified_linear_deriviative(val.z);
		val.w = soft_rectified_linear_deriviative(val.w);
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
		soft_rectified_linear_layer_updater_cuda::soft_rectified_linear_layer_updater_cuda()
		{
		}

		soft_rectified_linear_layer_updater_cuda::~soft_rectified_linear_layer_updater_cuda()
		{
		}

		void soft_rectified_linear_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("soft_rectified_linear_layer_updater_cuda is not able to run using offset");

			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			soft_rectified_linear_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				elem_count);
		}

		void soft_rectified_linear_layer_updater_cuda::enqueue_backprop(
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
			soft_rectified_linear_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				elem_count);
		}

		bool soft_rectified_linear_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
