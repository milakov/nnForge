/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "util_cuda.h"

#include "../rgb_to_yuv_convert_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

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
			float * __restrict errors,
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

				float y = errors[red_and_y_offset];
				float u = errors[green_and_u_offset];
				float v = errors[blue_and_v_offset];

				float red = y + reverse_r_v_mult * v;
				float green = y + reverse_g_u_mult * u + reverse_g_v_mult * v;
				float blue = y + reverse_b_u_mult * u;

				errors[red_and_y_offset] = red;
				errors[green_and_u_offset] = green;
				errors[blue_and_v_offset] = blue;
			}
		}

		rgb_to_yuv_convert_layer_updater_cuda::rgb_to_yuv_convert_layer_updater_cuda()
		{
		}

		rgb_to_yuv_convert_layer_updater_cuda::~rgb_to_yuv_convert_layer_updater_cuda()
		{
		}

		void rgb_to_yuv_convert_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("rgb_to_yuv_convert_layer_updater_cuda is not able to run using offset");

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
			rgb_to_yuv_convert_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				*schema_data[0],
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
		}

		void rgb_to_yuv_convert_layer_updater_cuda::enqueue_backprop(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
			rgb_to_yuv_convert_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*schema_data[0],
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				color_feature_map_config_count,
				entry_count);
		}

		bool rgb_to_yuv_convert_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}

		void rgb_to_yuv_convert_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const rgb_to_yuv_convert_layer> layer_derived = nnforge_dynamic_pointer_cast<const rgb_to_yuv_convert_layer>(layer_schema);

			color_feature_map_config_count = layer_derived->color_feature_map_config_list.size();
		}
	}
}
