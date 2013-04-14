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

#include "fully_connected_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "../convolution_layer.h"

__global__ void copy_bias_kernel(
	const float * __restrict biases,
	float * __restrict output,
	int output_neuron_count,
	int entry_count)
{
	int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

	if ((output_neuron_id < output_neuron_count))
	{
		float bias = biases[output_neuron_id];
		float * current_output = output + (int)(entry_id * output_neuron_count + output_neuron_id);
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

namespace nnforge
{
	namespace cuda
	{
		fully_connected_layer_tester_cuda::fully_connected_layer_tester_cuda()
		{
		}

		fully_connected_layer_tester_cuda::~fully_connected_layer_tester_cuda()
		{
		}

		void fully_connected_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_entry,
				(entry_count + 4 - 1) / 4,
				1);
			copy_bias_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*data[1],
				*additional_buffers[0],
				output_elem_count_per_entry,
				entry_count);

			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 1.0F;
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				output_elem_count_per_entry,
				entry_count,
				input_elem_count_per_entry,
				&alpha,
				*data[0],
				input_elem_count_per_entry,
				*input_buffer,
				input_elem_count_per_entry,
				&beta,
				*additional_buffers[0],
				output_elem_count_per_entry));

		}

		std::vector<size_t> fully_connected_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr fully_connected_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}
	}
}
