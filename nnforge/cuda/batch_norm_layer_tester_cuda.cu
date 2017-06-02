/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "batch_norm_layer_tester_cuda.h"

#include "neural_network_cudnn_exception.h"
#include "util_cuda.h"

#include "../batch_norm_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void batch_norm_kernel(
			float * __restrict output,
			const float * __restrict input,
			const float * __restrict gamma,
			const float * __restrict beta,
			const float * __restrict mean,
			const float * __restrict inverse_stddev,
			int elem_count_per_feature_map,
			int feature_map_count,
			int entry_count,
			int elem_count_per_entry)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int feature_map_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = (blockDim.z * blockIdx.z + threadIdx.z) * 2;
			if ((elem_id < elem_count_per_feature_map) && (feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				bool second_item_valid = (entry_id + 1 < entry_count);
				float mult = gamma[feature_map_id] * inverse_stddev[feature_map_id];
				float add = beta[feature_map_id] - mult * mean[feature_map_id];
				int offset1 = (entry_id * feature_map_count + feature_map_id) * elem_count_per_feature_map + elem_id;
				int offset2 = offset1 + elem_count_per_entry;
				float input_val1 = input[offset1];
				float input_val2;
				if (second_item_valid)
					input_val2 = input[offset2];
				float output_val1 = input_val1 * mult + add;
				float output_val2 = input_val2 * mult + add;
				output[offset1] = output_val1;
				if (second_item_valid)
					output[offset2] = output_val2;
			}
		}

		void batch_norm_layer_tester_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				(entry_count + 1) >> 1);
			batch_norm_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*data[0],
				*data[1],
				*data[2],
				*data[3],
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count,
				output_elem_count_per_entry);
		}

		void batch_norm_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const batch_norm_layer> layer_derived = std::dynamic_pointer_cast<const batch_norm_layer>(layer_schema);

			epsilon = layer_derived->epsilon;
			if (epsilon < CUDNN_BN_MIN_EPSILON)
				throw neural_network_exception((boost::format("Too small epsilon specified: %1%, cuDNN requires at least %2%") % epsilon % CUDNN_BN_MIN_EPSILON).str());
		}

		int batch_norm_layer_tester_cuda::get_input_index_layer_can_write() const
		{
			return 0;
		}
	}
}
