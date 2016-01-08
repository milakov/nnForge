/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "prefix_sum_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "../prefix_sum_layer.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];
		__global__ void prefix_sum_kernel(
			float * __restrict output,
			const float * __restrict input,
			int feature_map_segment_length,
			int neuron_count_per_entry,
			int neuron_count_per_feature_map,
			float clamp_min,
			float clamp_max,
			int iteration_count)
		{
			int threadblock_size = blockDim.x;
			int thread_id = threadIdx.x;
			int neuron_id = blockIdx.x;
			int feature_map_segment_id = blockIdx.y;
			int entry_id = blockIdx.z;

			int current_feature_map_local_id = thread_id;
			int offset = entry_id * neuron_count_per_entry + (feature_map_segment_id * feature_map_segment_length + current_feature_map_local_id) * neuron_count_per_feature_map + neuron_id;
			float running_sum = 0.0F;
			for(int i = 0; i < iteration_count; ++i, current_feature_map_local_id += threadblock_size)
			{
				float val = 0.0F;
				if (current_feature_map_local_id < feature_map_segment_length)
					val = input[offset];
				if (thread_id == 0)
					val += running_sum;

				arr_sh[thread_id] = val;

				__syncthreads();

				for(int d = 1; d < threadblock_size; d = d << 1)
				{
					if (thread_id >= d)
						val += arr_sh[thread_id - d];
					__syncthreads();
					if (thread_id >= d)
						arr_sh[thread_id] = val;
					__syncthreads();
				}

				if (thread_id == 0)
					running_sum = arr_sh[threadblock_size - 1];

				__syncthreads();

				if (current_feature_map_local_id < feature_map_segment_length)
					output[offset] = min(max(val, clamp_min), clamp_max);

				offset += threadblock_size * neuron_count_per_feature_map;
			}
		}

		prefix_sum_layer_tester_cuda::prefix_sum_layer_tester_cuda()
		{
		}

		prefix_sum_layer_tester_cuda::~prefix_sum_layer_tester_cuda()
		{
		}

		void prefix_sum_layer_tester_cuda::enqueue_forward_propagation(
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
			int threadblock_size = get_threadblock_size(feature_map_segment_length);
			int smem_size = threadblock_size * sizeof(float);
			int feature_map_segment_count = output_configuration_specific.feature_map_count / feature_map_segment_length;
			int iteration_count = (feature_map_segment_length + threadblock_size - 1) / threadblock_size;

			prefix_sum_kernel<<<dim3(output_elem_count_per_feature_map, feature_map_segment_count, entry_count), threadblock_size, smem_size, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				feature_map_segment_length,
				output_elem_count_per_entry,
				output_elem_count_per_feature_map,
				clamp_min,
				clamp_max,
				iteration_count);
		}

		void prefix_sum_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const prefix_sum_layer> layer_derived = nnforge_dynamic_pointer_cast<const prefix_sum_layer>(layer_schema);

			feature_map_segment_length = layer_derived->feature_map_segment_length;
			clamp_min = layer_derived->clamp_min;
			clamp_max = layer_derived->clamp_max;
		}

		int prefix_sum_layer_tester_cuda::get_threadblock_size(int feature_map_segment_length)
		{
			int threadblock_size;

			if (feature_map_segment_length < 256)
			{
				threadblock_size = (feature_map_segment_length + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (feature_map_segment_length + 256 - 1) / 256;
				threadblock_size = (feature_map_segment_length + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}

		int prefix_sum_layer_tester_cuda::get_input_index_layer_can_write() const
		{
			return 0;
		}
	}
}
