/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "rectified_linear_layer_updater_cuda.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void rectified_linear_upd_kernel(
			const float4 * __restrict input,
			float4 * __restrict output,
			uint4 * __restrict bits_buffer,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input[elem_id];
				uint4 bits;
				bits.x = __ballot(val.x > 0.0F ? 1 : 0);
				bits.y = __ballot(val.y > 0.0F ? 1 : 0);
				bits.z = __ballot(val.z > 0.0F ? 1 : 0);
				bits.w = __ballot(val.w > 0.0F ? 1 : 0);
				int lane_id = elem_id & 31;
				if (lane_id == 0)
					bits_buffer[elem_id >> 5] = bits;
				val.x = max(val.x, 0.0F);
				val.y = max(val.y, 0.0F);
				val.z = max(val.z, 0.0F);
				val.w = max(val.w, 0.0F);
				output[elem_id] = val;
			}
		}

		__global__ void rectified_linear_backprop_upd_kernel(
			float4 * __restrict input_errors,
			const float4 * __restrict output_errors,
			const uint4 * __restrict bits_buffer,
			bool add_update_to_destination,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = output_errors[elem_id];
				uint4 bits = bits_buffer[elem_id >> 5];
				int lane_id = elem_id & 31;
				unsigned int mask = (1 << lane_id);
				if ((bits.x & mask) == 0)
					val.x = 0.0F;
				if ((bits.y & mask) == 0)
					val.y = 0.0F;
				if ((bits.z & mask) == 0)
					val.z = 0.0F;
				if ((bits.w & mask) == 0)
					val.w = 0.0F;
				if (add_update_to_destination)
				{
					float4 prv = input_errors[elem_id];
					val.x += prv.x;
					val.y += prv.y;
					val.z += prv.z;
					val.w += prv.w;
				}
				input_errors[elem_id] = val;
			}
		}

		rectified_linear_layer_updater_cuda::rectified_linear_layer_updater_cuda()
		{
		}

		rectified_linear_layer_updater_cuda::~rectified_linear_layer_updater_cuda()
		{
		}

		void rectified_linear_layer_updater_cuda::enqueue_forward_propagation(
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
				elem_count,
				1,
				1,
				32);

			rectified_linear_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffers[0],
				*output_buffer,
				*temporary_per_entry_buffer,
				elem_count);
		}

		void rectified_linear_layer_updater_cuda::enqueue_backward_data_propagation(
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
				elem_count,
				1,
				1,
				32);
			rectified_linear_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_errors_buffer,
				*output_errors_buffer,
				*temporary_per_entry_buffer,
				add_update_to_destination,
				elem_count);
		}

		int rectified_linear_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool rectified_linear_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool rectified_linear_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool rectified_linear_layer_updater_cuda::is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const
		{
			return true;
		}

		size_t rectified_linear_layer_updater_cuda::get_temporary_per_entry_buffer_size() const
		{
			return ((output_elem_count_per_entry + 128 - 1) / 128) * sizeof(uint4);
		}
	}
}
