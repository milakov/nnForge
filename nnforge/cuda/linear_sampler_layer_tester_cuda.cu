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

#include "linear_sampler_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "../linear_sampler_layer.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void linear_sampler_2d_kernel(
			float * __restrict output,
			const float * __restrict grid,
			const float * __restrict input_data,
			int output_width,
			int output_height,
			int entry_count,
			int input_width,
			int input_height,
			int input_feature_map_count,
			float denormalize_scale_x,
			float denormalize_scale_y,
			int output_elem_count_per_feature_map,
			int input_elem_count_per_feature_map,
			int output_elem_count_per_entry,
			int input_elem_count_per_entry)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			if ((x < output_width) && (y < output_height) && (entry_id < entry_count))
			{
				int grid_x_offset = entry_id * output_elem_count_per_feature_map * 2 + y * output_width + x;
				int grid_y_offset = grid_x_offset + output_elem_count_per_feature_map;
				float normalized_x_pos = __load_nc(grid + grid_x_offset);
				float normalized_y_pos = __load_nc(grid + grid_y_offset);
				float absolute_x_pos = normalized_x_pos * denormalize_scale_x;
				float absolute_y_pos = normalized_y_pos * denormalize_scale_y;
				int left_x = __float2int_rd(absolute_x_pos);
				int top_y = __float2int_rd(absolute_y_pos);
				int right_x = left_x + 1;
				int bottom_y = top_y + 1;
				float right_weight = absolute_x_pos - (float)left_x;
				float left_weight = 1.0F - right_weight;
				float bottom_weight = absolute_y_pos - (float)top_y;
				float top_weight = 1.0F - bottom_weight;
				float top_left_weight = top_weight * left_weight;
				float top_right_weight = top_weight * right_weight;
				float bottom_left_weight = bottom_weight * left_weight;
				float bottom_right_weight = bottom_weight * right_weight;
				bool left_in_bounds = (unsigned int)left_x < (unsigned int)input_width;
				bool right_in_bounds = (unsigned int)right_x < (unsigned int)input_width;
				bool top_in_bounds = (unsigned int)top_y < (unsigned int)input_height;
				bool bottom_in_bounds = (unsigned int)bottom_y < (unsigned int)input_height;
				bool top_left_in_bounds = left_in_bounds && top_in_bounds;
				bool top_right_in_bounds = right_in_bounds && top_in_bounds;
				bool bottom_left_in_bounds = left_in_bounds && bottom_in_bounds;
				bool bottom_right_in_bounds = right_in_bounds && bottom_in_bounds;
				const float * current_input_data = input_data + entry_id * input_elem_count_per_entry + top_y * input_width + left_x;
				float * current_output = output + entry_id * output_elem_count_per_entry + y * output_width + x;
				for(int input_feature_map_id = 0; input_feature_map_id < input_feature_map_count; ++input_feature_map_id)
				{
					float top_left_val = top_left_in_bounds ? __load_nc(current_input_data) : 0.0F;
					float top_right_val = top_right_in_bounds ? __load_nc(current_input_data + 1) : 0.0F;
					float bottom_left_val = bottom_left_in_bounds ? __load_nc(current_input_data + input_width) : 0.0F;
					float bottom_right_val = bottom_right_in_bounds ? __load_nc(current_input_data + input_width + 1) : 0.0F;

					float weighted_sum = top_left_weight * top_left_val + top_right_weight * top_right_val + bottom_left_weight * bottom_left_val + bottom_right_weight * bottom_right_val;
					*current_output = weighted_sum;

					current_input_data += input_elem_count_per_feature_map;
					current_output += output_elem_count_per_feature_map;
				}
			}
		}

		void linear_sampler_layer_tester_cuda::enqueue_forward_propagation(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				entry_count);

			linear_sampler_2d_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*input_buffers[1],
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				entry_count,
				input_configuration_specific_list[1].dimension_sizes[0],
				input_configuration_specific_list[1].dimension_sizes[1],
				input_configuration_specific_list[1].feature_map_count,
				static_cast<float>(input_configuration_specific_list[1].dimension_sizes[0] - 1),
				static_cast<float>(input_configuration_specific_list[1].dimension_sizes[1] - 1),
				output_elem_count_per_feature_map,
				input_elem_count_per_feature_map_list[1],
				output_elem_count_per_entry,
				input_elem_count_per_entry_list[1]);
		}
	}
}
