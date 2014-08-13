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

#include "softmax_layer_hessian_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];

		__global__ void softmax_hess_kernel(
			const float * __restrict input,
			float * __restrict output,
			int feature_map_count,
			int neuron_count_per_feature_map,
			int entry_count)
		{
			int start_feature_map_id = threadIdx.x;
			int neuron_id = blockIdx.x;
			int entry_id = blockIdx.y;
			int threadblock_size = blockDim.x;

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;

		#if __CUDA_ARCH__ < 300
			volatile float * arr2 = arr_sh;
		#endif

			const float * start_input = input + (int)((entry_id * feature_map_count + start_feature_map_id) * neuron_count_per_feature_map + neuron_id);
			float * start_output = output + (int)((entry_id * feature_map_count + start_feature_map_id) * neuron_count_per_feature_map + neuron_id);
			unsigned int input_step = threadblock_size * neuron_count_per_feature_map;

			// calculate max value
			float max_value;
			{
				max_value = -1.0e+37F;
				const float * current_input = start_input;
				for(int feature_map_id = start_feature_map_id; feature_map_id < feature_map_count; feature_map_id += threadblock_size, current_input += input_step)
					max_value = max(max_value, __load_nc(current_input));
			#if __CUDA_ARCH__ < 300
				arr2[thread_id] = max_value;
			#endif
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
				#if __CUDA_ARCH__ < 300
					if (lane_id < tx)
						arr2[thread_id] = max(arr2[thread_id], arr2[thread_id + tx]);
				#else
					max_value = max(max_value, __shfl_down(max_value, tx));
				#endif
				}
			#if __CUDA_ARCH__ < 300
				max_value = arr2[thread_id];
				__syncthreads();
			#endif
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = max_value;
				__syncthreads();

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						max_value = max(max_value, arr_sh[i]);
					arr_sh[0] = max_value;
				}
				__syncthreads();

				max_value = arr_sh[0];
			}

			// calculate multiplier
			float mult;
			{
				float predicted_sum = 0.0F;
				const float * current_input = start_input;
				for(int feature_map_id = start_feature_map_id; feature_map_id < feature_map_count; feature_map_id += threadblock_size, current_input += input_step)
					predicted_sum += __expf(__load_nc(current_input) - max_value);

			#if __CUDA_ARCH__ < 300
				arr2[thread_id] = predicted_sum;
			#endif
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
				#if __CUDA_ARCH__ < 300
					if (lane_id < tx)
						arr2[thread_id] += arr2[thread_id + tx];
				#else
					predicted_sum += __shfl_down(predicted_sum, tx);
				#endif
				}
			#if __CUDA_ARCH__ < 300
				predicted_sum = arr2[thread_id];
				__syncthreads();
			#endif

				if (lane_id == 0)
					arr_sh[thread_id >> 5] = predicted_sum;
				__syncthreads();

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						predicted_sum += arr_sh[i];
					arr_sh[0] = __fdividef(1.0F, predicted_sum);
				}
				__syncthreads();

				mult = arr_sh[0];
			}

			// calculate error and gradient
			{
				const float * current_input = start_input;
				float * current_output = start_output;
				for(int feature_map_id = start_feature_map_id; feature_map_id < feature_map_count; feature_map_id += threadblock_size, current_input += input_step, current_output += input_step)
				{
					float val = __expf(__load_nc(current_input) - max_value);
					*current_output = val * mult;
				}
			}
		}

		__global__ void softmax_square_deriviative_hess_kernel(
			float * __restrict errors,
			const float * __restrict output_neurons,
			int feature_map_count,
			int neuron_count_per_feature_map,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
			if ((neuron_id < neuron_count_per_feature_map) && (entry_id < entry_count))
			{
				int initial_offset = entry_id * feature_map_count * neuron_count_per_feature_map + neuron_id;
				float sum = 0.0F;
				const float * current_output_neurons = output_neurons + initial_offset;
				const float * current_output_errors = errors + initial_offset;
				for(int i = 0; i < feature_map_count; ++i)
				{
					float y = __load_nc(current_output_neurons);
					sum += y * y * __load_nc(current_output_errors);
					current_output_neurons += neuron_count_per_feature_map;
					current_output_errors += neuron_count_per_feature_map;
				}

				current_output_neurons = output_neurons + initial_offset;
				float * current_errors = errors + initial_offset;
				for(int i = 0; i < feature_map_count; ++i)
				{
					float y = __load_nc(current_output_neurons);
					float y2 = y * y;
					*current_errors = y2 * (__load_nc(current_errors) * (2.0F * (y2 - y) + 1.0F) - sum);
					current_output_neurons += neuron_count_per_feature_map;
					current_errors += neuron_count_per_feature_map;
				}
			}
		}

		softmax_layer_hessian_cuda::softmax_layer_hessian_cuda()
		{
		}

		softmax_layer_hessian_cuda::~softmax_layer_hessian_cuda()
		{
		}

		void softmax_layer_hessian_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			int threadblock_size = get_threadblock_size(input_configuration_specific.feature_map_count);
			dim3 grid_size(input_elem_count_per_feature_map, entry_count, 1);
			dim3 block_size(threadblock_size, 1, 1);

			int smem_size = threadblock_size * sizeof(float);
			softmax_hess_kernel<<<grid_size, block_size, smem_size, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count);
		}

		void softmax_layer_hessian_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				entry_count,
				1);

			softmax_square_deriviative_hess_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count);
		}

		bool softmax_layer_hessian_cuda::is_in_place_backprop() const
		{
			return true;
		}

		int softmax_layer_hessian_cuda::get_threadblock_size(int output_neuron_count)
		{
			int threadblock_size;

			if (output_neuron_count < 256)
			{
				threadblock_size = (output_neuron_count + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (output_neuron_count + 256 - 1) / 256;
				threadblock_size = (output_neuron_count + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}
	}
}
