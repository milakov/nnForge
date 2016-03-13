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

#include "dropout_layer_updater_cuda.h"

#include "../dropout_layer.h"
#include "neural_network_curand_exception.h"
#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void dropout_upd_kernel(
			const float4 * __restrict input,
			float4 * __restrict output,
			const float4 * __restrict uniform_random, // (0.0,1.0]
			float mult,
			float keep_rate,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input[elem_id];
				float4 rnd = uniform_random[elem_id];
				val.x *= (rnd.x <= keep_rate ? mult : 0.0F);
				val.y *= (rnd.y <= keep_rate ? mult : 0.0F);
				val.z *= (rnd.z <= keep_rate ? mult : 0.0F);
				val.w *= (rnd.w <= keep_rate ? mult : 0.0F);
				output[elem_id] = val;
			}
		}

		__global__ void dropout_backprop_upd_kernel(
			float4 * __restrict input_errors,
			const float4 * __restrict output_errors,
			const float4 * __restrict uniform_random, // (0.0,1.0]
			float mult,
			float keep_rate,
			bool add_update_to_destination,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = output_errors[elem_id];
				float4 rnd = uniform_random[elem_id];
				float4 dst;
				if (add_update_to_destination)
				{
					dst = input_errors[elem_id];
					dst.x += val.x * (rnd.x <= keep_rate ? mult : 0.0F);
					dst.y += val.y * (rnd.y <= keep_rate ? mult : 0.0F);
					dst.z += val.z * (rnd.z <= keep_rate ? mult : 0.0F);
					dst.w += val.w * (rnd.w <= keep_rate ? mult : 0.0F);
				}
				else
				{
					dst.x = val.x * (rnd.x <= keep_rate ? mult : 0.0F);
					dst.y = val.y * (rnd.y <= keep_rate ? mult : 0.0F);
					dst.z = val.z * (rnd.z <= keep_rate ? mult : 0.0F);
					dst.w = val.w * (rnd.w <= keep_rate ? mult : 0.0F);
				}
				input_errors[elem_id] = dst;
			}
		}

		dropout_layer_updater_cuda::dropout_layer_updater_cuda()
		{
		}

		dropout_layer_updater_cuda::~dropout_layer_updater_cuda()
		{
		}

		void dropout_layer_updater_cuda::enqueue_forward_propagation(
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

			curand_safe_call(curandSetStream(cuda_config->get_curand_generator(), stream_id));
			curand_safe_call(curandGenerateUniform(cuda_config->get_curand_generator(), *temporary_per_entry_buffer, elem_count * 4));

			dropout_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffers[0],
				*output_buffer,
				*temporary_per_entry_buffer,
				mult,
				keep_rate,
				elem_count);
		}

		void dropout_layer_updater_cuda::enqueue_backward_data_propagation(
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
			dropout_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_errors_buffer,
				*output_errors_buffer,
				*temporary_per_entry_buffer,
				mult,
				keep_rate,
				add_update_to_destination,
				elem_count);
		}

		void dropout_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
			dropout_rate = layer_derived->dropout_rate;
			keep_rate = 1.0F - dropout_rate;
			mult = 1.0F / keep_rate;
		}

		int dropout_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool dropout_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool dropout_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool dropout_layer_updater_cuda::is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const
		{
			return true;
		}

		size_t dropout_layer_updater_cuda::get_temporary_per_entry_buffer_size() const
		{
			return output_elem_count_per_entry * sizeof(float);
		}
	}
}
