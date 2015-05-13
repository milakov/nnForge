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

#include "sigmoid_partial_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../sigmoid_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void sigmoid_upd_kernel(
			const float * __restrict input,
			float * __restrict output,
			const int * __restrict affected_feature_map_list,
			int feature_map_count,
			int elem_count_per_feature_map,
			int affected_feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int affected_feature_map_config_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (affected_feature_map_config_id < affected_feature_map_count) && (entry_id < entry_count))
			{
				int feature_map_id = affected_feature_map_list[affected_feature_map_config_id];

				int offset = (entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map +  elem_id;

				float val = input[offset];
				float new_val = __fdividef(1.0F, 1.0F + __expf(-val));
				output[offset] = new_val;
			}
		}

		__global__ void sigmoid_backprop_upd_kernel(
			float * __restrict errors,
			const float * __restrict output_neurons,
			const int * __restrict affected_feature_map_list,
			int feature_map_count,
			int elem_count_per_feature_map,
			int affected_feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int affected_feature_map_config_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (affected_feature_map_config_id < affected_feature_map_count) && (entry_id < entry_count))
			{
				int feature_map_id = affected_feature_map_list[affected_feature_map_config_id];

				int offset = (entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map +  elem_id;

				float current_error = errors[offset];
				float out_neuron = output_neurons[offset];
				float der1st = out_neuron * (1.0F - out_neuron);
				errors[offset] = current_error * der1st;
			}
		}

		sigmoid_partial_layer_updater_cuda::sigmoid_partial_layer_updater_cuda()
		{
		}

		sigmoid_partial_layer_updater_cuda::~sigmoid_partial_layer_updater_cuda()
		{
		}

		void sigmoid_partial_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("sigmoid_partial_layer_updater_cuda is not able to run using offset");

			cuda_util::copy_buffer(
				*cuda_config,
				*input_neurons_buffer,
				*output_neurons_buffer,
				entry_count * input_elem_count_per_entry,
				stream_id);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				affected_feature_map_count,
				entry_count);
			sigmoid_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				*schema_data[0],
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				affected_feature_map_count,
				entry_count);
		}

		void sigmoid_partial_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				affected_feature_map_count,
				entry_count);
			sigmoid_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				*schema_data[0],
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				affected_feature_map_count,
				entry_count);
		}

		bool sigmoid_partial_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}

		void sigmoid_partial_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const sigmoid_layer> layer_derived = nnforge_dynamic_pointer_cast<const sigmoid_layer>(layer_schema);

			affected_feature_map_count = static_cast<int>(layer_derived->affected_feature_map_id_list.size());
		}
	}
}
