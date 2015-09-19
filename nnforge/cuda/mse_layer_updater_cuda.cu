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

#include "mse_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "../mse_layer.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];
		__global__ void mse_upd_kernel(
			float * __restrict output,
			const float * __restrict input0,
			const float * __restrict input1,
			int input_feature_map_count,
			int elem_count_per_feature_map,
			float scale,
			int entry_count)
		{
			int feature_map_id = threadIdx.x;
			int neuron_id = blockIdx.x;
			int entry_id = blockIdx.y;
			int threadblock_size = blockDim.x;

			int input_offset = (entry_id * input_feature_map_count + feature_map_id) * elem_count_per_feature_map + neuron_id;
			float err = 0.0F;
			while (feature_map_id < input_feature_map_count)
			{
				float local_err = input0[input_offset] - input1[input_offset];
				err += local_err * local_err;
				feature_map_id += threadblock_size;
				input_offset += threadblock_size * elem_count_per_feature_map;
			}

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
				err += __shfl_down(err, tx);

			int warp_count = threadblock_size >> 5;
			if (warp_count > 1)
			{
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = err;

				__syncthreads();

				if (thread_id < 32)
				{
					err = 0.0F;
					if (thread_id < warp_count)
						err = arr_sh[thread_id];
					#pragma unroll
					for(int tx = 4; tx > 0; tx >>= 1)
						err += __shfl_down(err, tx);
				}
			}
		
			if (thread_id == 0)
				output[entry_id * elem_count_per_feature_map + neuron_id] = err * scale;
		}

		template<bool add_update_to_destination>
		__global__ void mse_backprop_upd_kernel(
			float * __restrict output,
			const float * __restrict deriv_input_neurons,
			const float * __restrict target_input_neurons,
			float scale2,
			int elem_count) 
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				if (add_update_to_destination)
					output[elem_id] += scale2 * (target_input_neurons[elem_id] - deriv_input_neurons[elem_id]);
				else
					output[elem_id] = scale2 * (target_input_neurons[elem_id] - deriv_input_neurons[elem_id]);
			}
		}

		mse_layer_updater_cuda::mse_layer_updater_cuda()
		{
		}

		mse_layer_updater_cuda::~mse_layer_updater_cuda()
		{
		}

		void mse_layer_updater_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			int threadblock_size = get_threadblock_size(input_configuration_specific_list[0].feature_map_count);
			int elem_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();

			int smem_size = ((threadblock_size + 32 - 1) / 32) * sizeof(float);
			mse_upd_kernel<<<dim3(elem_count_per_feature_map, entry_count), threadblock_size, smem_size, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*input_buffers[1],
				input_configuration_specific_list[0].feature_map_count,
				elem_count_per_feature_map,
				scale,
				entry_count);
		}

		void mse_layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			int elem_count = entry_count * input_elem_count_per_entry_list[0];
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);

			if (add_update_to_destination)
				mse_backprop_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*input_neurons_buffers[input_index],
					*input_neurons_buffers[1 - input_index],
					scale * 2.0F,
					elem_count);
			else
				mse_backprop_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*input_neurons_buffers[input_index],
					*input_neurons_buffers[1 - input_index],
					scale * 2.0F,
					elem_count);
		}

		void mse_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const mse_layer> layer_derived = nnforge_dynamic_pointer_cast<const mse_layer>(layer_schema);

			scale = layer_derived->scale;
		}

		bool mse_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return true;
		}

		bool mse_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		int mse_layer_updater_cuda::get_threadblock_size(int input_feature_map_count)
		{
			int threadblock_size;

			if (input_feature_map_count < 256)
			{
				threadblock_size = (input_feature_map_count + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (input_feature_map_count + 256 - 1) / 256;
				threadblock_size = (input_feature_map_count + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}
	}
}
