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

#include "sigmoid_partial_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../sigmoid_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void sigmoid_kernel(
			float * __restrict input,
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
				input[offset] = new_val;
			}
		}

		sigmoid_partial_layer_tester_cuda::sigmoid_partial_layer_tester_cuda()
		{
		}

		sigmoid_partial_layer_tester_cuda::~sigmoid_partial_layer_tester_cuda()
		{
		}

		void sigmoid_partial_layer_tester_cuda::enqueue_test(
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
				affected_feature_map_count,
				entry_count);
			sigmoid_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_buffer,
				*schema_data[0],
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				affected_feature_map_count,
				entry_count);
		}

		void sigmoid_partial_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const sigmoid_layer> layer_derived = nnforge_dynamic_pointer_cast<const sigmoid_layer>(layer_schema);

			affected_feature_map_count = static_cast<int>(layer_derived->affected_feature_map_id_list.size());
		}
	}
}
