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

#include "lerror_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "../lerror_layer.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];
		template<int n_type>
		__global__ void lerror_kernel(
			float * __restrict output,
			const float * __restrict input0,
			const float * __restrict input1,
			const float * __restrict scale_mask,
			int input_feature_map_count,
			int elem_count_per_feature_map,
			float n_value,
			float scale,
			int entry_count)
		{
			int feature_map_id = threadIdx.x;
			int neuron_id = blockIdx.x;
			int entry_id = blockIdx.y;
			int threadblock_size = blockDim.x;

			float err = 0.0F;
			int output_offset = entry_id * elem_count_per_feature_map + neuron_id;

			float mask = 1.0F;
			if (scale_mask)
				mask = scale_mask[output_offset];

			int thread_id = threadIdx.x;
			if (mask != 0.0F)
			{
				int input_offset = (entry_id * input_feature_map_count + feature_map_id) * elem_count_per_feature_map + neuron_id;
				while (feature_map_id < input_feature_map_count)
				{
					float local_err = input0[input_offset] - input1[input_offset];

					if (n_type == 1)
						err += fabsf(local_err);
					else if (n_type == 2)
						err += local_err * local_err;
					else
						err += __powf(fabsf(local_err), n_value);

					feature_map_id += threadblock_size;
					input_offset += threadblock_size * elem_count_per_feature_map;
				}

				int lane_id = thread_id & 31;
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
					err += __shfl_down(err, tx);

				int warp_count = threadblock_size >> 5;
				if (warp_count > 1)
				{
					if (lane_id == 0)
						arr_sh[thread_id >> 5] = err;

					__syncthreads();

					if (thread_id < 32)
					{
						err = 0.0F;
						if (thread_id < warp_count)
							err = arr_sh[thread_id];
						#pragma unroll
						for(int tx = 4; tx > 0; tx >>= 1)
							err += __shfl_down(err, tx);
					}
				}
			}
		
			if (thread_id == 0)
				output[output_offset] = err * (mask * scale);
		}

		void lerror_layer_tester_cuda::enqueue_forward_propagation(
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
			int threadblock_size = get_threadblock_size(input_configuration_specific_list[0].feature_map_count);

			const float * scale_mask = 0;
			if (input_buffers.size() > 2)
				scale_mask = *input_buffers[2];

			int smem_size = ((threadblock_size + 32 - 1) / 32) * sizeof(float);
			if (n_value == 1.0F)
				lerror_kernel<1><<<dim3(input_elem_count_per_feature_map_list[0], entry_count), threadblock_size, smem_size, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					*input_buffers[1],
					scale_mask,
					input_configuration_specific_list[0].feature_map_count,
					input_elem_count_per_feature_map_list[0],
					n_value,
					scale,
					entry_count);
			else if (n_value == 2.0F)
				lerror_kernel<2><<<dim3(input_elem_count_per_feature_map_list[0], entry_count), threadblock_size, smem_size, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					*input_buffers[1],
					scale_mask,
					input_configuration_specific_list[0].feature_map_count,
					input_elem_count_per_feature_map_list[0],
					n_value,
					scale,
					entry_count);
			else
				lerror_kernel<-1><<<dim3(input_elem_count_per_feature_map_list[0], entry_count), threadblock_size, smem_size, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					*input_buffers[1],
					scale_mask,
					input_configuration_specific_list[0].feature_map_count,
					input_elem_count_per_feature_map_list[0],
					n_value,
					scale,
					entry_count);
		}

		void lerror_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const lerror_layer> layer_derived = std::dynamic_pointer_cast<const lerror_layer>(layer_schema);

			scale = layer_derived->scale;
			n_value = layer_derived->n;
		}

		int lerror_layer_tester_cuda::get_threadblock_size(int input_feature_map_count)
		{
			int threadblock_size;

			if (input_feature_map_count < 256)
			{
				threadblock_size = (input_feature_map_count + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (input_feature_map_count + 256 - 1) / 256;
				threadblock_size = (input_feature_map_count + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}
	}
}
