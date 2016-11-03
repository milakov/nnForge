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

#include "rgb_to_yuv_convert_layer_updater_cuda.h"

#include <cuda_runtime.h>
#include <memory>

#include "util_cuda.h"

#include "../rgb_to_yuv_convert_layer.h"
#include "../neural_network_exception.h"

#define w_r 0.299F
#define w_b 0.114F
#define w_g (1.0F - w_r - w_b)
#define u_max 0.436F
#define v_max 0.615F
#define u_mult (u_max / (1.0F - w_b))
#define v_mult (v_max / (1.0F - w_r))

#define reverse_r_v_mult ((1.0F - w_r) / v_max)
#define reverse_g_u_mult (-(w_b * (1.0F - w_b)) / (u_max * w_g))
#define reverse_g_v_mult (-(w_r * (1.0F - w_r)) / (v_max * w_g))
#define reverse_b_u_mult ((1.0F - w_b) / u_max)

namespace nnforge
{
	namespace cuda
	{
		__global__ void rgb_to_yuv_convert_upd_kernel(
			const float * __restrict input,
			float * __restrict output,
			const int * __restrict color_feature_map_config_list,
			int feature_map_count,
			int elem_count_per_feature_map,
			int color_feature_map_config_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int color_feature_map_config_config_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (color_feature_map_config_config_id < color_feature_map_config_count) && (entry_id < entry_count))
			{
				int color_feature_map_config_id_offset = color_feature_map_config_config_id * 3;
				int red_and_y_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset];
				int green_and_u_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset + 1];
				int blue_and_v_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset + 2];

				int base_offset = (entry_id * elem_count_per_feature_map * feature_map_count) + elem_id;
				int red_and_y_offset = red_and_y_feature_map_id * elem_count_per_feature_map + base_offset;
				int green_and_u_offset = green_and_u_feature_map_id * elem_count_per_feature_map + base_offset;
				int blue_and_v_offset = blue_and_v_feature_map_id * elem_count_per_feature_map + base_offset;

				float red = input[red_and_y_offset];
				float green = input[green_and_u_offset];
				float blue = input[blue_and_v_offset];

				float y = w_r * red + w_g * green + w_b * blue;
				float u = u_mult * (blue - y);
				float v = v_mult * (red - y);

				output[red_and_y_offset] = y;
				output[green_and_u_offset] = u;
				output[blue_and_v_offset] = v;
			}
		}

		__global__ void rgb_to_yuv_convert_deriviative_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
			const int * __restrict color_feature_map_config_list,
			int feature_map_count,
			int elem_count_per_feature_map,
			int color_feature_map_config_count,
			bool add_update_to_destination,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int color_feature_map_config_config_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (color_feature_map_config_config_id < color_feature_map_config_count) && (entry_id < entry_count))
			{
				int color_feature_map_config_id_offset = color_feature_map_config_config_id * 3;
				int red_and_y_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset];
				int green_and_u_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset + 1];
				int blue_and_v_feature_map_id = color_feature_map_config_list[color_feature_map_config_id_offset + 2];

				int base_offset = (entry_id * elem_count_per_feature_map * feature_map_count) + elem_id;
				int red_and_y_offset = red_and_y_feature_map_id * elem_count_per_feature_map + base_offset;
				int green_and_u_offset = green_and_u_feature_map_id * elem_count_per_feature_map + base_offset;
				int blue_and_v_offset = blue_and_v_feature_map_id * elem_count_per_feature_map + base_offset;

				float y = output_errors[red_and_y_offset];
				float u = output_errors[green_and_u_offset];
				float v = output_errors[blue_and_v_offset];

				float red = y + reverse_r_v_mult * v;
				float green = y + reverse_g_u_mult * u + reverse_g_v_mult * v;
				float blue = y + reverse_b_u_mult * u;

				if (add_update_to_destination)
				{
					input_errors[red_and_y_offset] += red;
					input_errors[green_and_u_offset] += green;
					input_errors[blue_and_v_offset] += blue;
				}
				else
				{
					input_errors[red_and_y_offset] = red;
					input_errors[green_and_u_offset] = green;
					input_errors[blue_and_v_offset] = blue;
				}
			}
		}

		void rgb_to_yuv_convert_layer_updater_cuda::enqueue_forward_propagation(
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
			if ((color_feature_map_config_count != output_configuration_specific.feature_map_count * 3) && ((const float *)*output_buffer != (const float *)*input_buffers[1]))
			{
				cuda_util::copy_buffer(
					*cuda_config,
					*input_buffers[0],
					*output_buffer,
					output_elem_count_per_entry * entry_count,
					stream_id);
			}

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
			rgb_to_yuv_convert_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffers[0],
				*output_buffer,
				*schema_data[0],
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
		}

		void rgb_to_yuv_convert_layer_updater_cuda::enqueue_backward_data_propagation(
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
			if (((const float *)*output_errors_buffer != (const float *)*input_errors_buffer)
				&& ((color_feature_map_config_count != output_configuration_specific.feature_map_count * 3) || add_update_to_destination))
			{
				cuda_util::copy_buffer(
					*cuda_config,
					*output_errors_buffer,
					*input_errors_buffer,
					output_elem_count_per_entry * entry_count,
					stream_id);
			}

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
			rgb_to_yuv_convert_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_errors_buffer,
				*output_errors_buffer,
				*schema_data[0],
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				color_feature_map_config_count,
				add_update_to_destination,
				entry_count);
		}

		int rgb_to_yuv_convert_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool rgb_to_yuv_convert_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool rgb_to_yuv_convert_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		void rgb_to_yuv_convert_layer_updater_cuda::updater_configured()
		{
			std::shared_ptr<const rgb_to_yuv_convert_layer> layer_derived = std::dynamic_pointer_cast<const rgb_to_yuv_convert_layer>(layer_schema);

			color_feature_map_config_count = static_cast<int>(layer_derived->color_feature_map_config_list.size());
		}
	}
}
