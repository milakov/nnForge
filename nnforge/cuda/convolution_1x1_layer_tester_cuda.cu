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

#include "convolution_1x1_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void copy_bias_1x1_kernel(
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

		convolution_1x1_layer_tester_cuda::convolution_1x1_layer_tester_cuda()
		{
		}

		convolution_1x1_layer_tester_cuda::~convolution_1x1_layer_tester_cuda()
		{
		}

		void convolution_1x1_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				(entry_count + 4 - 1) / 4,
				1);
			copy_bias_1x1_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*data[1],
				*additional_buffers[2],
				output_elem_count_per_entry,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);

			cuda_util::transpose(
				*cuda_config,
				*input_buffer,
				*additional_buffers[1],
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
				*additional_buffers[1],
				input_configuration_specific.feature_map_count,
				&beta,
				*additional_buffers[2],
				output_configuration_specific.feature_map_count));

			cuda_util::transpose(
				*cuda_config,
				*additional_buffers[2],
				*additional_buffers[0],
				output_configuration_specific.feature_map_count,
				output_elem_count_per_feature_map,
				entry_count,
				stream_id);
		}

		std::vector<size_t> convolution_1x1_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));
			res.push_back(input_elem_count_per_entry * sizeof(float));
			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr convolution_1x1_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		int convolution_1x1_layer_tester_cuda::get_bias_update_block_size(int entry_count)
		{
			int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
