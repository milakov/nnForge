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

#include "sigmoid_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../nn_types.h"

static __forceinline__ __device__ float sigmoid(float x)
{
	return __fdividef(1.0F, 1.0F + __expf(-x));
}

__global__ void sigmoid_kernel(
	float4 * __restrict input,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = input[elem_id];
		val.x = sigmoid(val.x);
		val.y = sigmoid(val.y);
		val.z = sigmoid(val.z);
		val.w = sigmoid(val.w);
		input[elem_id] = val;
	}
}

namespace nnforge
{
	namespace cuda
	{
		sigmoid_layer_tester_cuda::sigmoid_layer_tester_cuda()
		{
		}

		sigmoid_layer_tester_cuda::~sigmoid_layer_tester_cuda()
		{
		}

		void sigmoid_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			sigmoid_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffer,
				elem_count);
		}
	}
}
