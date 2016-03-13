/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "add_layer_updater_cuda.h"

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "../add_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void add_upd_kernel(
			float4 * __restrict output_buf,
			const float4 * __restrict input_buf1,
			const float4 * __restrict input_buf2,
			float alpha1,
			float alpha2,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val1 = input_buf1[elem_id];
				float4 val2 = input_buf2[elem_id];
				val1.x = val1.x * alpha1 + val2.x * alpha2;
				val1.y = val1.y * alpha1 + val2.y * alpha2;
				val1.z = val1.z * alpha1 + val2.z * alpha2;
				val1.w = val1.w * alpha1 + val2.w * alpha2;
				output_buf[elem_id] = val1;
			}
		}

		add_layer_updater_cuda::add_layer_updater_cuda()
		{
		}

		add_layer_updater_cuda::~add_layer_updater_cuda()
		{
		}

		void add_layer_updater_cuda::enqueue_forward_propagation(
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
			unsigned int elem_count = output_elem_count_per_entry * entry_count;
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				new_elem_count);

			add_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*input_buffers[1],
				alpha,
				alpha,
				new_elem_count);

			for(int i = 2; i < static_cast<int>(input_configuration_specific_list.size()); ++i)
			{
				add_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*output_buffer,
					*input_buffers[i],
					1.0F,
					alpha,
					new_elem_count);
			}
		}

		void add_layer_updater_cuda::enqueue_backward_data_propagation(
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
			unsigned int elem_count = output_elem_count_per_entry * entry_count;

			cuda_util::multiply_by_value(
				*cuda_config,
				*input_errors_buffer,
				*output_errors_buffer,
				alpha,
				elem_count,
				add_update_to_destination,
				stream_id);
		}

		int add_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool add_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool add_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		void add_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const add_layer> layer_derived = nnforge_dynamic_pointer_cast<const add_layer>(layer_schema);

			alpha = layer_derived->alpha;
		}
	}
}
