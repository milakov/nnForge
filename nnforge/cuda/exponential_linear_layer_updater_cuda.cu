/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "exponential_linear_layer_updater_cuda.h"

#include <cuda_runtime.h>
#include <memory>

#include "../neural_network_exception.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		static __forceinline__ __device__ float exponential_linear(float x)
		{
			return (x >= 0.0F) ? x : (__expf(x) - 1.0F);
		}

		__global__ void exponential_linear_upd_kernel(
			const float4 * __restrict input,
			float4 * __restrict output,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input[elem_id];
				val.x = exponential_linear(val.x);
				val.y = exponential_linear(val.y);
				val.z = exponential_linear(val.z);
				val.w = exponential_linear(val.w);
				output[elem_id] = val;
			}
		}

		static __forceinline__ __device__ float exponential_linear_deriviative(float x)
		{
			return (x >= 0.0F) ? 1.0F : (x + 1.0F);
		}

		__global__ void hyperbolic_tangent_deriviative_upd_kernel(
			float4 * __restrict input_errors,
			const float4 * __restrict output_errors,
			const float4 * __restrict output_neurons,
			bool add_update_to_destination,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = output_neurons[elem_id];
				val.x = exponential_linear_deriviative(val.x);
				val.y = exponential_linear_deriviative(val.y);
				val.z = exponential_linear_deriviative(val.z);
				val.w = exponential_linear_deriviative(val.w);
				float4 current_error = output_errors[elem_id];
				float4 current_dst;
				if (add_update_to_destination)
				{
					current_dst = input_errors[elem_id];
					current_dst.x += current_error.x * val.x;
					current_dst.y += current_error.y * val.y;
					current_dst.z += current_error.z * val.z;
					current_dst.w += current_error.w * val.w;
				}
				else
				{
					current_dst.x = current_error.x * val.x;
					current_dst.y = current_error.y * val.y;
					current_dst.z = current_error.z * val.z;
					current_dst.w = current_error.w * val.w;
				}
				input_errors[elem_id] = current_dst;
			}
		}

		void exponential_linear_layer_updater_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			int elem_count = (output_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			exponential_linear_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffers[0],
				*output_buffer,
				elem_count);
		}

		void exponential_linear_layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			int elem_count = (output_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			hyperbolic_tangent_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_errors_buffer,
				*output_errors_buffer,
				*output_neurons_buffer,
				add_update_to_destination,
				elem_count);
		}

		int exponential_linear_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool exponential_linear_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool exponential_linear_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return true;
		}
	}
}
