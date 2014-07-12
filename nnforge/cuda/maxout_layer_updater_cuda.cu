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

#include "maxout_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../maxout_layer.h"
#include "../nn_types.h"

__global__ void maxout_upd_kernel(
	float * __restrict output,
	int * __restrict max_feature_map_positions,
	const float * __restrict input,
	int neuron_count_per_feature_map,
	int input_feature_map_count,
	int output_feature_map_count,
	int feature_map_subsampling_size,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	if ((neuron_id < neuron_count_per_feature_map) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int input_offset = (entry_id * input_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		float max_val = input[input_offset];
		int max_pos = 0;
		for(int i = 1; i < feature_map_subsampling_size; ++i)
		{
			input_offset += output_feature_map_count * neuron_count_per_feature_map;
			float new_val = input[input_offset];
			if (new_val > max_val)
			{
				max_val = new_val;
				max_pos = i;
			}
		}
		int output_offset = (entry_id * output_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		output[output_offset] = max_val;
		max_feature_map_positions[output_offset] = max_pos;
	}
}

__global__ void maxout_deriviative_upd_kernel(
	float * __restrict input_errors,
	const int * __restrict max_feature_map_positions,
	const float * __restrict output_errors,
	int neuron_count_per_feature_map,
	int input_feature_map_count,
	int output_feature_map_count,
	int feature_map_subsampling_size,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	if ((neuron_id < neuron_count_per_feature_map) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int output_offset = (entry_id * output_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		int max_feature_map = max_feature_map_positions[output_offset];
		float output_error = output_errors[output_offset];

		int input_offset = (entry_id * input_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		for(int i = 0; i < feature_map_subsampling_size; ++i)
		{
			input_errors[input_offset] = ((i == max_feature_map) ? output_error : 0.0F);
			input_offset += output_feature_map_count * neuron_count_per_feature_map;
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		maxout_layer_updater_cuda::maxout_layer_updater_cuda()
		{
		}

		maxout_layer_updater_cuda::~maxout_layer_updater_cuda()
		{
		}

		void maxout_layer_updater_cuda::enqueue_test(
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
				throw neural_network_exception("maxout_layer_updater_cuda is not able to run using offset");

			const float * input = *input_neurons_buffer;
			float * output = *output_neurons_buffer;
			int * max_feature_map_positions = *additional_buffers[0];

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);

			maxout_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				output,
				max_feature_map_positions,
				input,
				output_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				output_configuration_specific.feature_map_count,
				feature_map_subsampling_size,
				entry_count);
		}

		void maxout_layer_updater_cuda::enqueue_backprop(
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
			const float * output_errors = *output_errors_buffer;
			int * max_feature_map_positions = *additional_buffers[0];
			float * input_errors = *input_errors_buffer;

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);

			maxout_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				input_errors,
				max_feature_map_positions,
				output_errors,
				output_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				output_configuration_specific.feature_map_count,
				feature_map_subsampling_size,
				entry_count);
		}

		void maxout_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const maxout_layer> layer_derived = nnforge_dynamic_pointer_cast<const maxout_layer>(layer_schema);

			feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
		}

		bool maxout_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		std::vector<size_t> maxout_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}
	}
}
