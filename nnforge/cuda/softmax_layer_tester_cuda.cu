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

#include "softmax_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../softmax_layer.h"

__global__ void softmax_kernel(
	float * __restrict input,
	int feature_map_count,
	int neuron_count_per_feature_map,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	if ((neuron_id < neuron_count_per_feature_map) && (entry_id < entry_count))
	{
		float * initial_input = input + entry_id * feature_map_count * neuron_count_per_feature_map + neuron_id;
		float sum = 0.0F;
		float * current_input = initial_input;
		for(int i = 0; i < feature_map_count; ++i)
		{
			sum += __expf(__load_nc(current_input));
			current_input += neuron_count_per_feature_map;
		}

		float mult = __fdividef(1.0F, sum);
		current_input = initial_input;
		for(int i = 0; i < feature_map_count; ++i)
		{
			float val = __expf(__load_nc(current_input));
			*current_input = val * mult;
			current_input += neuron_count_per_feature_map;
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		softmax_layer_tester_cuda::softmax_layer_tester_cuda()
		{
		}

		softmax_layer_tester_cuda::~softmax_layer_tester_cuda()
		{
		}

		void softmax_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				entry_count,
				1);

			softmax_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count);
		}
	}
}
