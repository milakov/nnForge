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

#include "convolution_1x1_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void copy_bias_1x1_upd_kernel(
			const float * __restrict biases,
			float * __restrict output,
			int output_neuron_count,
			int output_neuron_count_per_feature_map,
			int output_feature_map_count,
			int entry_count)
		{
			int feature_map_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_neuron_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * 4;

			if ((feature_map_id < output_feature_map_count) && (output_neuron_id < output_neuron_count_per_feature_map) && (entry_id < entry_count))
			{
				float bias = biases[feature_map_id];
				float * current_output = output + (int)(entry_id * output_neuron_count + output_neuron_id * output_feature_map_count + feature_map_id);
				#pragma unroll
				for(int i = 0; i < 4; ++i)
				{
					if (entry_id < entry_count)
						*current_output = bias;
					current_output += output_neuron_count;
					entry_id++;
				}
			}
		}

		extern __shared__ float arr[];
		__global__ void convolution_1x1_update_biases_upd_kernel(
			float * __restrict gradient_biases,
			const float * __restrict output_errors,
			int block_size,
			int output_elem_count_per_feature_map,
			int output_feature_map_count,
			int entry_count)
		{
			int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = blockIdx.y;
			int block_id = blockIdx.z * blockDim.z + threadIdx.z;
			int base_entry_id = block_size * block_id;
			int thread_id = blockDim.x * threadIdx.z + threadIdx.x;
			int threadblock_size = blockDim.x * blockDim.z;
			float sum = 0.0F;
			int iteration_count = min(entry_count - base_entry_id, block_size);
			if (output_neuron_id < output_elem_count_per_feature_map)
			{
				const float * current_error = output_errors + (base_entry_id * output_feature_map_count + output_feature_map_id) * output_elem_count_per_feature_map + output_neuron_id;
				int output_elem_count_per_entry = output_elem_count_per_feature_map * output_feature_map_count;
				for(int i = 0; i < iteration_count; ++i)
				{
					sum += *current_error;
					current_error += output_elem_count_per_entry;
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
				atomicAdd(gradient_biases + output_feature_map_id, arr[0]);
		}

		convolution_1x1_layer_updater_cuda::convolution_1x1_layer_updater_cuda()
		{
		}

		convolution_1x1_layer_updater_cuda::~convolution_1x1_layer_updater_cuda()
		{
		}

		void convolution_1x1_layer_updater_cuda::enqueue_test(
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
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				(entry_count + 4 - 1) / 4,
				1);
			copy_bias_1x1_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*data[1],
				*additional_buffers[1],
				output_elem_count_per_entry,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);
			
			cuda_util::transpose(
				*cuda_config,
				(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
				*additional_buffers[0],
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count,
				stream_id);

			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 1.0F;
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				output_configuration_specific.feature_map_count,
				entry_count * input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				&alpha,
				*data[0],
				input_configuration_specific.feature_map_count,
				*additional_buffers[0],
				input_configuration_specific.feature_map_count,
				&beta,
				*additional_buffers[1],
				output_configuration_specific.feature_map_count));

			cuda_util::transpose(
				*cuda_config,
				*additional_buffers[1],
				*output_neurons_buffer,
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				entry_count,
				stream_id);
		}

		void convolution_1x1_layer_updater_cuda::enqueue_backprop(
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
			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 0.0F;
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				input_configuration_specific.feature_map_count,
				entry_count * input_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				&alpha,
				*data[0],
				input_configuration_specific.feature_map_count,
				*additional_buffers[1],
				output_configuration_specific.feature_map_count,
				&beta,
				*additional_buffers[0],
				input_configuration_specific.feature_map_count));

			cuda_util::transpose(
				*cuda_config,
				*additional_buffers[0],
				*input_errors_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count,
				stream_id);
		}

		void convolution_1x1_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			// Update biases
			{
				int block_size = get_bias_update_block_size(entry_count);
				int block_count = (entry_count + block_size - 1) / block_size;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					output_elem_count_per_feature_map,
					1,
					block_count);
				kernel_dims.first.y = output_configuration_specific.feature_map_count;
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				convolution_1x1_update_biases_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*gradient[1],
					*output_errors_buffer,
					block_size,
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count);
			}
		
			// Update weights
			{
				cuda_util::transpose(
					*cuda_config,
					*output_errors_buffer,
					*additional_buffers[1],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 1.0F;
				cublas_safe_call(cublasSgemm(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					input_configuration_specific.feature_map_count,
					output_configuration_specific.feature_map_count,
					entry_count * input_elem_count_per_feature_map,
					&alpha,
					*additional_buffers[0],
					input_configuration_specific.feature_map_count,
					*additional_buffers[1],
					output_configuration_specific.feature_map_count,
					&beta,
					*gradient[0],
					input_configuration_specific.feature_map_count));
			}
		}

		std::vector<size_t> convolution_1x1_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_entry * sizeof(float));
			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		bool convolution_1x1_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		int convolution_1x1_layer_updater_cuda::get_bias_update_block_size(int entry_count)
		{
			int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
