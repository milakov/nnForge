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

#include "affine_grid_generator_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "../affine_grid_generator_layer.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		template <bool adjust_for_zero_init>
		__global__ void affine_grid_generator_2d_kernel(
			float * __restrict output,
			const float2 * __restrict input,
			int width,
			int height,
			int entry_count,
			float x_scale,
			float y_scale,
			float weight_scale,
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
				#pragma unroll
				for(int i = 0; i < 3; ++i)
				{
					input_vals[i] = __load_nc(current_input + i);
					input_vals[i].x *= weight_scale;
					input_vals[i].y *= weight_scale;
				}
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

		void affine_grid_generator_layer_tester_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				entry_count);

			if (adjust_for_zero_init)
				affine_grid_generator_2d_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					entry_count,
					x_scale,
					y_scale,
					weight_scale,
					output_elem_count_per_feature_map);
			else
				affine_grid_generator_2d_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_configuration_specific.dimension_sizes[0],
					output_configuration_specific.dimension_sizes[1],
					entry_count,
					x_scale,
					y_scale,
					weight_scale,
					output_elem_count_per_feature_map);
		}

		void affine_grid_generator_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const affine_grid_generator_layer> layer_derived = std::dynamic_pointer_cast<const affine_grid_generator_layer>(layer_schema);

			adjust_for_zero_init = layer_derived->adjust_for_zero_init;

			x_scale = output_configuration_specific.dimension_sizes[0] > 1 ? 1.0F / static_cast<float>(output_configuration_specific.dimension_sizes[0] - 1) : 1.0F;
			y_scale = output_configuration_specific.dimension_sizes[1] > 1 ? 1.0F / static_cast<float>(output_configuration_specific.dimension_sizes[1] - 1) : 1.0F;
			weight_scale = layer_derived->get_weight_scale(output_configuration_specific);
		}
	}
}
