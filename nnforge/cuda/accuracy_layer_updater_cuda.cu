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

#include "accuracy_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../accuracy_layer.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];
		__global__ void accuracy_upd_kernel(
			float * __restrict output,
			const float * __restrict predicted,
			const float * __restrict actual,
			const float * __restrict scale_mask,
			int input_feature_map_count,
			int elem_count_per_feature_map,
			int output_elem_count_per_entry,
			unsigned int top_n,
			int entry_count)
		{
			int start_feature_map_id = threadIdx.x;
			int neuron_id = blockIdx.x;
			int entry_id = blockIdx.y;
			int threadblock_size = blockDim.x;

			float mask = 1.0F;
			if (scale_mask)
				mask = scale_mask[entry_id * elem_count_per_feature_map + neuron_id];

			int sum = 0;
			int thread_id = threadIdx.x;
			if (mask != 0.0F)
			{
				int start_input_offset = (entry_id * input_feature_map_count + start_feature_map_id) * elem_count_per_feature_map + neuron_id;
				int max_val_feature_map_id = -1;
				float max_val = -1.0e37F;

				int warp_count = threadblock_size >> 5;
				float * val_sh = arr_sh;
				int * fm_sh = (int *)(arr_sh + warp_count);
				int * cnt_sh = (int *)(arr_sh + 2 * warp_count);

				int lane_id = thread_id & 31;

				int input_offset = start_input_offset;
				int feature_map_id = start_feature_map_id;
				while (feature_map_id < input_feature_map_count)
				{
					float new_val = actual[input_offset];
					if (new_val > max_val)
					{
						max_val = new_val;
						max_val_feature_map_id = feature_map_id;
					}
					feature_map_id += threadblock_size;
					input_offset += threadblock_size * elem_count_per_feature_map;
				}

				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
					float new_val = __shfl_down(max_val, tx);
					int feature_map_id = __shfl_down(max_val_feature_map_id, tx);
#else
					float new_val = __shfl_down_sync(0xFFFFFFFF, max_val, tx);
					int feature_map_id = __shfl_down_sync(0xFFFFFFFF, max_val_feature_map_id, tx);
#endif
#endif

					if ((new_val > max_val) || ((new_val == max_val) && (feature_map_id < max_val_feature_map_id)))
					{
						max_val = new_val;
						max_val_feature_map_id = feature_map_id;
					}
				}

				if (warp_count > 1)
				{
					if (lane_id == 0)
					{
						val_sh[thread_id >> 5] = max_val;
						fm_sh[thread_id >> 5] = max_val_feature_map_id;
					}

					__syncthreads();

					if (thread_id < 32)
					{
						max_val = -1.0e37F;
						max_val_feature_map_id = -1;
						if (thread_id < warp_count)
						{
							max_val = val_sh[thread_id];
							max_val_feature_map_id = fm_sh[thread_id];
						}
						#pragma unroll
						for(int tx = 4; tx > 0; tx >>= 1)
						{
#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
							float new_val = __shfl_down(max_val, tx);
							int feature_map_id = __shfl_down(max_val_feature_map_id, tx);
#else
							float new_val = __shfl_down_sync(0xFFFFFFFF, max_val, tx);
							int feature_map_id = __shfl_down_sync(0xFFFFFFFF, max_val_feature_map_id, tx);
#endif
#endif

							if ((new_val > max_val) || ((new_val == max_val) && (feature_map_id < max_val_feature_map_id)))
							{
								max_val = new_val;
								max_val_feature_map_id = feature_map_id;
							}
						}
					}

					if (thread_id == 0)
					{
						val_sh[0] = predicted[(entry_id * input_feature_map_count + max_val_feature_map_id) * elem_count_per_feature_map + neuron_id];
						fm_sh[0] = max_val_feature_map_id;
					}

					__syncthreads();

					max_val = val_sh[0];
					max_val_feature_map_id = fm_sh[0];
				} // if (warp_count > 1)
				else
				{
					if (thread_id == 0)
						max_val = predicted[(entry_id * input_feature_map_count + max_val_feature_map_id) * elem_count_per_feature_map + neuron_id];

#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
					max_val_feature_map_id = __shfl(max_val_feature_map_id, 0);
					max_val = __shfl(max_val, 0);
#else
					max_val_feature_map_id = __shfl_sync(0xFFFFFFFF, max_val_feature_map_id, 0);
					max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
#endif
#endif
				}

				// max_val and max_val_feature_map_id set for all threads
				// Writing to val_sh and fm_sh is not safe here yet

				sum = 0;
				input_offset = start_input_offset;
				feature_map_id = start_feature_map_id;
				while (feature_map_id < input_feature_map_count)
				{
					float val = predicted[input_offset];
					if ((val > max_val) || ((val == max_val) && (feature_map_id < max_val_feature_map_id)))
						++sum;
					feature_map_id += threadblock_size;
					input_offset += threadblock_size * elem_count_per_feature_map;
				}

				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
					sum += __shfl_down(sum, tx);
#else
					sum += __shfl_down_sync(0xFFFFFFFF, sum, tx);
#endif
#endif

				if (warp_count > 1)
				{
					if (lane_id == 0)
						cnt_sh[thread_id >> 5] = sum;

					__syncthreads();

					if (thread_id < 32)
					{
						sum = 0;
						if (thread_id < warp_count)
							sum = cnt_sh[thread_id];
						#pragma unroll
						for(int tx = 4; tx > 0; tx >>= 1)
#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
							sum += __shfl_down(sum, tx);
#else
							sum += __shfl_down_sync(0xFFFFFFFF, sum, tx);
#endif
#endif
					}
				}
			}

			if (thread_id == 0)
			{
				int output_offset = entry_id * output_elem_count_per_entry + neuron_id;
				for(int i = 0; i < top_n; ++i)
				{
					output[output_offset] = ((sum <= i) ? mask : 0.0F);
					output_offset += elem_count_per_feature_map;
				}
				output[output_offset] = mask; // Scale
			}
		}

		void accuracy_layer_updater_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			int threadblock_size = get_threadblock_size(input_configuration_specific_list[0].feature_map_count);

			const float * scale_mask = 0;
			if (input_buffers.size() > 2)
				scale_mask = *input_buffers[2];

			int smem_size = ((threadblock_size + 32 - 1) / 32) * (sizeof(float) + 2 * sizeof(int));
			accuracy_upd_kernel<<<dim3(input_elem_count_per_feature_map_list[0], entry_count), threadblock_size, smem_size, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				*input_buffers[1],
				scale_mask,
				input_configuration_specific_list[0].feature_map_count,
				input_elem_count_per_feature_map_list[0],
				output_elem_count_per_entry,
				top_n,
				entry_count);
		}

		void accuracy_layer_updater_cuda::updater_configured()
		{
			std::shared_ptr<const accuracy_layer> layer_derived = std::dynamic_pointer_cast<const accuracy_layer>(layer_schema);

			top_n = layer_derived->top_n;
		}

		int accuracy_layer_updater_cuda::get_threadblock_size(int input_feature_map_count)
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
