/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "max_subsampling_2d_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../max_subsampling_layer.h"
#include "../nn_types.h"

struct __align__(4) window_x_x_config
{
	window_x_x_config(int window_x, int x)
	{
		this->window_x_x_pair = (((unsigned int)window_x) << 16) | (unsigned int)x;
	}

	unsigned int window_x_x_pair;
};

struct __align__(4) y_feature_map_config
{
	y_feature_map_config(int y, int feature_map_id)
	{
		this->y_feature_map_id_pair = (((unsigned int)y) << 16) | (unsigned int)feature_map_id;
	}

	unsigned int y_feature_map_id_pair;
};

extern __shared__ float arr_sh[];

__global__ void max_subsampling_2d_tex_kernel(
	float * __restrict output,
	const float * __restrict input,
	const window_x_x_config * __restrict window_x_x_config_list,
	const y_feature_map_config * __restrict y_feature_map_config_list,
	int subsampling_width,
	int subsampling_height,
	int input_width,
	int input_height,
	int output_width,
	int output_height,
	int feature_map_count,
	int entry_count,
	int window_x_x_config_count,
	int y_feature_map_config_count)
{
	int window_x_x_config_id = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_config_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	int local_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	float * vals = arr_sh;

	bool in_bounds = (entry_id < entry_count) && (window_x_x_config_id < window_x_x_config_count) && (feature_map_config_id < y_feature_map_config_count);

	float res = -1.0e37F;
	int window_x;
	int output_x;
	int output_y;
	int feature_map_id;
	if (in_bounds)
	{
		window_x_x_config wxx = window_x_x_config_list[window_x_x_config_id];
		output_x = wxx.window_x_x_pair & 0xFFFF;
		window_x = wxx.window_x_x_pair >> 16;

		y_feature_map_config yfm = y_feature_map_config_list[feature_map_config_id];
		feature_map_id = yfm.y_feature_map_id_pair & 0xFFFF;
		output_y = yfm.y_feature_map_id_pair >> 16;

		int input_x = output_x * subsampling_width + window_x;
		int input_y = output_y * subsampling_height;

		int current_input_elem_id = ((entry_id * feature_map_count + feature_map_id) * input_height + input_y) * input_width + input_x;

		res = input[current_input_elem_id];
		for(int j = 1; j < subsampling_height; ++j)
		{
			current_input_elem_id += input_width;
			res = max(res, input[current_input_elem_id]);
		}

		vals[local_thread_id] = res;
	}

	__syncthreads();

	if (in_bounds && (window_x == 0))
	{
		for(int j = 1; j < subsampling_width; ++j)
		{
			local_thread_id++;
			res = max(res, vals[local_thread_id]);
		}
		output[((entry_id * feature_map_count + feature_map_id) * output_height + output_y) * output_width + output_x] = res;
	}
}

namespace nnforge
{
	namespace cuda
	{
		max_subsampling_2d_layer_tester_cuda::max_subsampling_2d_layer_tester_cuda()
		{
		}

		max_subsampling_2d_layer_tester_cuda::~max_subsampling_2d_layer_tester_cuda()
		{
		}

		void max_subsampling_2d_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			const float * input = *input_buffer;
			float * output = *additional_buffers[0];

			int window_x_x_config_count = subsampling_sizes[0] * output_configuration_specific.dimension_sizes[0];
			const window_x_x_config * window_x_x_config_list = static_cast<const window_x_x_config *>((const void *)*additional_buffers[1]);

			int y_feature_map_config_count = output_configuration_specific.dimension_sizes[1] * output_configuration_specific.feature_map_count;
			const y_feature_map_config * y_feature_map_config_list = static_cast<const y_feature_map_config *>((const void *)*additional_buffers[2]);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				window_x_x_config_count,
				y_feature_map_config_count,
				entry_count,
				subsampling_sizes[0]);

			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = threadblock_size * sizeof(float);

			max_subsampling_2d_tex_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
				output,
				input,
				window_x_x_config_list,
				y_feature_map_config_list,
				subsampling_sizes[0],
				subsampling_sizes[1],
				input_configuration_specific.dimension_sizes[0],
				input_configuration_specific.dimension_sizes[1],
				output_configuration_specific.dimension_sizes[0],
				output_configuration_specific.dimension_sizes[1],
				output_configuration_specific.feature_map_count,
				entry_count,
				window_x_x_config_count,
				y_feature_map_config_count);
		}

		std::vector<size_t> max_subsampling_2d_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr max_subsampling_2d_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		void max_subsampling_2d_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

			subsampling_sizes = layer_derived->subsampling_sizes;
		}

		std::vector<size_t> max_subsampling_2d_layer_tester_cuda::get_sizes_of_additional_buffers_fixed() const
		{
			std::vector<size_t> res;

			res.push_back(sizeof(window_x_x_config) * subsampling_sizes[0] * output_configuration_specific.dimension_sizes[0]);
			res.push_back(sizeof(y_feature_map_config) * output_configuration_specific.dimension_sizes[1] * output_configuration_specific.feature_map_count);

			return res;
		}

		void max_subsampling_2d_layer_tester_cuda::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
			{
				std::vector<window_x_x_config> task_list;
				for(int x = 0; x < output_configuration_specific.dimension_sizes[0]; ++x)
					for(int window_x = 0; window_x < subsampling_sizes[0]; ++window_x)
						task_list.push_back(window_x_x_config(window_x, x));

				cuda_safe_call(cudaMemcpy(*additional_buffers[1], &(*task_list.begin()), sizeof(window_x_x_config) * task_list.size(), cudaMemcpyHostToDevice));
			}

			{
				std::vector<y_feature_map_config> task_list;
				for(int feature_map_id = 0; feature_map_id < output_configuration_specific.feature_map_count; ++feature_map_id)
					for(int y = 0; y < output_configuration_specific.dimension_sizes[1]; ++y)
						task_list.push_back(y_feature_map_config(y, feature_map_id));

				cuda_safe_call(cudaMemcpy(*additional_buffers[2], &(*task_list.begin()), sizeof(y_feature_map_config) * task_list.size(), cudaMemcpyHostToDevice));
			}
		}
	}
}
