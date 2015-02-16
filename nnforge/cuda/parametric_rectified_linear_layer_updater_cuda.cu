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

#include "parametric_rectified_linear_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../parametric_rectified_linear_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void parametric_rectified_linear_upd_kernel(
			float * __restrict output,
			const float * __restrict input,
			const float * __restrict data,
			int elem_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int feature_map_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				float a = __load_nc(data + feature_map_id);
				int offset = (entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map + elem_id;
				float input_val = input[offset];
				float output_val = input_val * (input_val >= 0.0F ? 1.0F : a);
				output[offset] = output_val;
			}
		}

		__global__ void parametric_rectified_linear_backprop_upd_kernel(
			float * __restrict errors,
			const float * __restrict input_neurons,
			const float * __restrict data,
			int elem_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int feature_map_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id < elem_count_per_feature_map) && (feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				float a = __load_nc(data + feature_map_id);
				int offset = (entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map + elem_id;
				float output_err = errors[offset];
				float input_val = input_neurons[offset];
				float input_err = output_err * (input_val >= 0.0F ? 1.0F : a);
				errors[offset] = input_err;
			}
		}

		extern __shared__ float arr[];
		__global__ void parametric_rectified_linear_update_weights_upd_kernel(
			float * __restrict gradients,
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			int block_size,
			int elem_count_per_feature_map,
			int elem_count_per_entry,
			int feature_map_count,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int feature_map_id = blockIdx.y;
			int block_id = blockIdx.z * blockDim.z + threadIdx.z;
			int base_entry_id = block_size * block_id;
			int thread_id = blockDim.x * threadIdx.z + threadIdx.x;
			int threadblock_size = blockDim.x * blockDim.z;
			float sum = 0.0F;
			int iteration_count = min(entry_count - base_entry_id, block_size);
			if (neuron_id < elem_count_per_feature_map)
			{
				int offset = (base_entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map + neuron_id;
				const float * current_error = output_errors + offset;
				const float * current_neurons = input_neurons + offset;
				for(int i = 0; i < iteration_count; ++i)
				{
					float output_err = *current_error;
					float input_val = *current_neurons;
					float gr = output_err * (input_val >= 0.0F ? 0.0F : input_val);
					sum += gr;
					current_error += elem_count_per_entry;
					current_neurons += elem_count_per_entry;
				}
			}
			arr[thread_id] = sum;
			__syncthreads();

			int t_add_elems = threadblock_size >> 1;
			int t_working_elems = (threadblock_size + 1) >> 1;
			while (t_add_elems > 0)
			{
				if (thread_id < t_add_elems)
					arr[thread_id] += arr[thread_id + t_working_elems];
				t_add_elems = t_working_elems >> 1;
				t_working_elems = (t_working_elems + 1) >> 1;
				__syncthreads();
			}

			if (thread_id == 0)
				atomicAdd(gradients + feature_map_id, arr[0]);
		}

		parametric_rectified_linear_layer_updater_cuda::parametric_rectified_linear_layer_updater_cuda()
		{
		}

		parametric_rectified_linear_layer_updater_cuda::~parametric_rectified_linear_layer_updater_cuda()
		{
		}

		void parametric_rectified_linear_layer_updater_cuda::enqueue_test(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
			parametric_rectified_linear_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_neurons_buffer,
				(const float *)(*input_neurons_buffer) + offset_input_entry_id * input_elem_count_per_entry,
				*data[0],
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
		}

		void parametric_rectified_linear_layer_updater_cuda::enqueue_backprop(
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
				input_configuration_specific.feature_map_count,
				entry_count);
			parametric_rectified_linear_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*input_neurons_buffer,
				*data[0],
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
		}

		void parametric_rectified_linear_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			int block_size = get_update_block_size(entry_count);
			int block_count = (entry_count + block_size - 1) / block_size;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				1,
				block_count);
			kernel_dims.first.y = output_configuration_specific.feature_map_count;
			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = threadblock_size * sizeof(float);
			parametric_rectified_linear_update_weights_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
				*gradient[0],
				*output_errors_buffer,
				(const float *)(*input_neurons_buffer) + offset_input_entry_id * input_elem_count_per_entry,
				block_size,
				output_elem_count_per_feature_map,
				input_elem_count_per_entry,
				output_configuration_specific.feature_map_count,
				entry_count);
		}

		bool parametric_rectified_linear_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}

		int parametric_rectified_linear_layer_updater_cuda::get_update_block_size(int entry_count)
		{
			int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
