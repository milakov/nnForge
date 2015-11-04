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

		template<bool add_update_to_destination>
		__global__ void parametric_rectified_linear_backprop_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
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
				float output_err = output_errors[offset];
				float input_val = input_neurons[offset];
				float input_err = output_err * (input_val >= 0.0F ? 1.0F : a);
				if (add_update_to_destination)
					input_errors[offset] += input_err;
				else
					input_errors[offset] = input_err;
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

		void parametric_rectified_linear_layer_updater_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);
			parametric_rectified_linear_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*data[0],
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);
		}

		void parametric_rectified_linear_layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);
			if (add_update_to_destination)
				parametric_rectified_linear_backprop_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
					*input_neurons_buffers[0],
					*data[0],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count);
			else
				parametric_rectified_linear_backprop_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
					*input_neurons_buffers[0],
					*data[0],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count);
		}

		void parametric_rectified_linear_layer_updater_cuda::enqueue_backward_weights_propagation(
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& gradient,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			unsigned int entry_count)
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
				*input_neurons_buffers[0],
				block_size,
				output_elem_count_per_feature_map,
				output_elem_count_per_entry,
				output_configuration_specific.feature_map_count,
				entry_count);
		}

		int parametric_rectified_linear_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool parametric_rectified_linear_layer_updater_cuda::is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			return true;
		}

		bool parametric_rectified_linear_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return true;
		}

		bool parametric_rectified_linear_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		int parametric_rectified_linear_layer_updater_cuda::get_update_block_size(int entry_count)
		{
			int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
