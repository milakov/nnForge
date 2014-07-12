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

#include "absolute_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../neural_network_exception.h"
#include "util_cuda.h"


__global__ void absolute_upd_kernel(
	const float4 * __restrict input,
	float4 * __restrict output,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = input[elem_id];
		val.x = fabsf(val.x);
		val.y = fabsf(val.y);
		val.z = fabsf(val.z);
		val.w = fabsf(val.w);
		output[elem_id] = val;
	}
}

__global__ void absolute_deriviative_upd_kernel(
	float4 * __restrict errors,
	const float4 * __restrict input_neurons,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 inp = input_neurons[elem_id];
		float4 current_error = errors[elem_id];
		if (inp.x < 0.0F)
			current_error.x = -current_error.x;
		if (inp.y < 0.0F)
			current_error.y = -current_error.y;
		if (inp.z < 0.0F)
			current_error.z = -current_error.z;
		if (inp.w < 0.0F)
			current_error.w = -current_error.w;
		errors[elem_id] = current_error;
	}
}

namespace nnforge
{
	namespace cuda
	{
		absolute_layer_updater_cuda::absolute_layer_updater_cuda()
		{
		}

		absolute_layer_updater_cuda::~absolute_layer_updater_cuda()
		{
		}

		void absolute_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("absolute_layer_updater_cuda is not able to run using offset");

			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			absolute_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				elem_count);
		}

		void absolute_layer_updater_cuda::enqueue_backprop(
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
			absolute_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*input_neurons_buffer,
				elem_count);
		}

		bool absolute_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
