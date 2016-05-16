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

#include "affine_grid_generator_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../affine_grid_generator_layer.h"
#include "../nn_types.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		template <bool adjust_for_zero_init>
		__global__ void affine_grid_generator_2d_upd_kernel(
			float * __restrict output,
			const float2 * __restrict input,
			int width,
			int height,
			int entry_count,
			float x_scale,
			float y_scale,
			int output_elem_count_per_feature_map)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			if ((x < width) && (y < height) && (entry_id < entry_count))
			{
				float x_pos = (float)x * x_scale;
				float y_pos = (float)y * y_scale;

				const float2 * current_input = input + entry_id * 3;
				float2 input_vals[3];
				for(int i = 0; i < 3; ++i)
					input_vals[i] = __load_nc(current_input + i);
				if (adjust_for_zero_init)
				{
					input_vals[0].x += 1.0F;
					input_vals[2].x += 1.0F;
				}

				float x_out_pos = x_pos * input_vals[0].x + y_pos * input_vals[0].y + input_vals[1].x;
				float y_out_pos = x_pos * input_vals[1].y + y_pos * input_vals[2].x + input_vals[2].y;

				int x_output_offset = (entry_id * height * 2 + y) * width + x;
				int y_output_offset = x_output_offset + output_elem_count_per_feature_map;

				output[x_output_offset] = x_out_pos;
				output[y_output_offset] = y_out_pos;
			}
		}

		extern __shared__ float arr_sh[];
		__global__ void affine_grid_generator_2d_backprop_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
			int width,
			int height,
			int entry_count,
			float x_scale,
			float y_scale,
			int output_elem_count_per_feature_map)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = ((x < width) && (y < height) && (entry_id < entry_count));
			int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
			int threadblock_size = blockDim.x * blockDim.y;

			int x_output_offset = (entry_id * height * 2 + y) * width + x;
			int y_output_offset = x_output_offset + output_elem_count_per_feature_map;
			float x_out_err = in_bounds ? output_errors[x_output_offset] : 0.0F;
			float y_out_err = in_bounds ? output_errors[y_output_offset] : 0.0F;
			float x_pos = (float)x * x_scale;
			float y_pos = (float)y * y_scale;

			float in_errors[6];
			in_errors[0] = x_pos * x_out_err;
			in_errors[1] = y_pos * x_out_err;
			in_errors[2] = x_out_err;
			in_errors[3] = x_pos * y_out_err;
			in_errors[4] = y_pos * y_out_err;
			in_errors[5] = y_out_err;

			int current_max_thread_id_with_data = threadblock_size;
			int max_transfer_count = threadblock_size >> 1;
			while (current_max_thread_id_with_data > 1)
			{
				int new_max_thread_id_with_data = (current_max_thread_id_with_data + 1) >> 1;

				if ((thread_id < current_max_thread_id_with_data) && (thread_id >= new_max_thread_id_with_data))
				{
					#pragma unroll
					for(int i = 0; i < 6; ++i)
						arr_sh[i * max_transfer_count + (thread_id - new_max_thread_id_with_data)] = in_errors[i];
				}

				__syncthreads();

				if (thread_id < (current_max_thread_id_with_data - new_max_thread_id_with_data))
				{
					#pragma unroll
					for(int i = 0; i < 6; ++i)
						in_errors[i] += arr_sh[i * max_transfer_count + thread_id];
				}

				current_max_thread_id_with_data = new_max_thread_id_with_data;

				__syncthreads();
			}

			if (in_bounds && (thread_id == 0))
			{
				float * current_input_errors = input_errors + entry_id * 6;
				#pragma unroll
				for(int i = 0; i < 6; ++i)
					atomicAdd(current_input_errors + i, in_errors[i]);
			}
		}

		affine_grid_generator_layer_updater_cuda::affine_grid_generator_layer_updater_cuda()
		{
		}

		affine_grid_generator_layer_updater_cuda::~affine_grid_generator_layer_updater_cuda()
		{
		}

		void affine_grid_generator_layer_updater_cuda::enqueue_forward_propagation(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				entry_count);

			if (adjust_for_zero_init)
				affine_grid_generator_2d_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					entry_count,
					x_scale,
					y_scale,
					output_elem_count_per_feature_map);
			else
				affine_grid_generator_2d_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					entry_count,
					x_scale,
					y_scale,
					output_elem_count_per_feature_map);
		}

		void affine_grid_generator_layer_updater_cuda::enqueue_backward_data_propagation(
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
			if (!add_update_to_destination)
			{
				cuda_util::set_with_value(
					*cuda_config,
					*input_errors_buffer,
					0.0F,
					input_elem_count_per_entry_list[0] * entry_count,
					stream_id);
			}

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				1);
			kernel_dims.first.z = entry_count;
			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = (threadblock_size >> 1) * 6 * sizeof(float);

			affine_grid_generator_2d_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
				*input_errors_buffer,
				*output_errors_buffer,
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				entry_count,
				x_scale,
				y_scale,
				output_elem_count_per_feature_map);
		}

		void affine_grid_generator_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const affine_grid_generator_layer> layer_derived = nnforge_dynamic_pointer_cast<const affine_grid_generator_layer>(layer_schema);

			adjust_for_zero_init = layer_derived->adjust_for_zero_init;

			x_scale = output_configuration_specific.dimension_sizes[0] > 1 ? 1.0F / static_cast<float>(output_configuration_specific.dimension_sizes[0] - 1) : 1.0F;
			y_scale = output_configuration_specific.dimension_sizes[1] > 1 ? 1.0F / static_cast<float>(output_configuration_specific.dimension_sizes[1] - 1) : 1.0F;
		}

		bool affine_grid_generator_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool affine_grid_generator_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}
	}
}
