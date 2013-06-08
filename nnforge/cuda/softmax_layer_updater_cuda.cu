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

#include "softmax_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../neural_network_exception.h"

#include "util_cuda.h"

__global__ void softmax_upd_kernel(
	const float * __restrict input,
	float * __restrict output,
	int feature_map_count,
	int neuron_count_per_feature_map,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	if ((neuron_id < neuron_count_per_feature_map) && (entry_id < entry_count))
	{
		int initial_offset = entry_id * feature_map_count * neuron_count_per_feature_map + neuron_id;
		float sum = 0.0F;
		const float * current_input = input + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			sum += __expf(*current_input);
			current_input += neuron_count_per_feature_map;
		}

		float mult = __fdividef(1.0F, sum);
		current_input = input + initial_offset;
		float * current_output = output + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			float val = __expf(*current_input);
			*current_output = val * mult;
			current_input += neuron_count_per_feature_map;
			current_output += neuron_count_per_feature_map;
		}
	}
}


__global__ void softmax_deriviative_upd_kernel(
	float * __restrict errors,
	const float * __restrict output_neurons,
	int feature_map_count,
	int neuron_count_per_feature_map,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	if ((neuron_id < neuron_count_per_feature_map) && (entry_id < entry_count))
	{
		int initial_offset = entry_id * feature_map_count * neuron_count_per_feature_map + neuron_id;
		float sum = 0.0F;
		const float * current_output_neurons = output_neurons + initial_offset;
		const float * current_output_errors = errors + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			sum += __load_nc(current_output_neurons) * __load_nc(current_output_errors);
			current_output_neurons += neuron_count_per_feature_map;
			current_output_errors += neuron_count_per_feature_map;
		}

		current_output_neurons = output_neurons + initial_offset;
		float * current_errors = errors + initial_offset;
		for(int i = 0; i < feature_map_count; ++i)
		{
			*current_errors = __load_nc(current_output_neurons) * (__load_nc(current_errors) - sum);
			current_output_neurons += neuron_count_per_feature_map;
			current_errors += neuron_count_per_feature_map;
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		softmax_layer_updater_cuda::softmax_layer_updater_cuda()
		{
		}

		softmax_layer_updater_cuda::~softmax_layer_updater_cuda()
		{
		}

		void softmax_layer_updater_cuda::enqueue_test(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				entry_count,
				1);

			softmax_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count);
		}

		void softmax_layer_updater_cuda::enqueue_backprop(
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
				entry_count,
				1);

			softmax_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count);
		}

		bool softmax_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}

		void softmax_layer_updater_cuda::updater_configured()
		{
			if (!different_input)
				throw neural_network_exception("softmax_layer_updater_cuda is not able to run using the same input");
		}
	}
}
