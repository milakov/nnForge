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

#include "parametric_rectified_linear_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../parametric_rectified_linear_layer.h"
#include "../nn_types.h"

__global__ void parametric_rectified_linear_kernel(
	float * __restrict input,
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
		input[offset] = output_val;
	}
}

namespace nnforge
{
	namespace cuda
	{
		parametric_rectified_linear_layer_tester_cuda::parametric_rectified_linear_layer_tester_cuda()
		{
		}

		parametric_rectified_linear_layer_tester_cuda::~parametric_rectified_linear_layer_tester_cuda()
		{
		}

		void parametric_rectified_linear_layer_tester_cuda::enqueue_test(
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
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
			parametric_rectified_linear_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffer,
				*data[0],
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
		}
	}
}
