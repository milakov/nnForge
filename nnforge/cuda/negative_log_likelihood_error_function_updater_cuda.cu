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

#include "negative_log_likelihood_error_function_updater_cuda.h"

#include "../negative_log_likelihood_error_function.h"
#include "../softmax_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__forceinline__ __device__ double atomicAdd(double* address, double val)
		{
				unsigned long long int* address_as_ull = (unsigned long long int*)address;
				unsigned long long int old = *address_as_ull, assumed;
				do {
					assumed = old;
					old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
				} while (assumed != old);
				return __longlong_as_double(old);
		}

		extern __shared__ float arr_sh[];

		__global__ void negative_log_likelihood_update_error_and_gradient_kernel(
			float * __restrict gradients,
			double * __restrict total_error,
			const float * __restrict actual_output_neurons,
			const float * __restrict predicted_output_neurons,
			int offset_entry_id,
			int neuron_count,
			int updater_entry_count)
		{
			int neuron_id = blockIdx.y * blockDim.x + threadIdx.x;
			int updater_entry_id = blockIdx.x;

			int offset = updater_entry_id * neuron_count + neuron_id;
			float err = 0.0F;
			if (neuron_id < neuron_count)
			{
				float actual_val = actual_output_neurons[(offset_entry_id + updater_entry_id) * neuron_count + neuron_id];
				float predicted_val = predicted_output_neurons[offset];
				err = (actual_val > 0.0F) ? - actual_val * __logf(max(predicted_val, 1.0e-20F)) : 0.0F;
				gradients[offset] = (actual_val > 0.0F) ? __fdividef(actual_val, predicted_val) : 0.0F;
			}

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;
		#if __CUDA_ARCH__ < 300
			volatile float * arr = arr_sh;
			arr[thread_id] = err;
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			#else
				err += __shfl_down(err, tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			err = arr[thread_id];
			__syncthreads();
		#endif

			if (blockDim.x > 32)
			{
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = err;
				__syncthreads();
			}

			if (thread_id == 0)
			{
				for(int i = 1; i < (blockDim.x >> 5); ++i)
					err += arr_sh[i];
				double err_d = (double)err;

				atomicAdd(total_error, err_d);
			}
		}

		__global__ void negative_log_likelihood_update_error_and_gradient_fused_with_activation_kernel(
			float * __restrict gradients,
			double * __restrict total_error,
			const float * __restrict actual_output_neurons,
			const float * __restrict predicted_output_neurons,
			int offset_entry_id,
			int neuron_count,
			int updater_entry_count)
		{
			int start_neuron_id = threadIdx.x;
			int updater_entry_id = blockIdx.x;
			int threadblock_size = blockDim.x;

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;

			float predicted_sum = 0.0F;
			for(int neuron_id = start_neuron_id; neuron_id < neuron_count; neuron_id += threadblock_size)
				predicted_sum += __expf(predicted_output_neurons[updater_entry_id * neuron_count + neuron_id]);

		#if __CUDA_ARCH__ < 300
			volatile float * arr2 = arr_sh;
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

			float mult = arr_sh[0];

			float err = 0.0F;
			for(int neuron_id = start_neuron_id; neuron_id < neuron_count; neuron_id += threadblock_size)
			{
				float actual_val = actual_output_neurons[(offset_entry_id + updater_entry_id) * neuron_count + neuron_id];
				float predicted_val = __expf(predicted_output_neurons[updater_entry_id * neuron_count + neuron_id]) * mult;
				gradients[updater_entry_id * neuron_count + neuron_id] = actual_val - predicted_val;
				if (actual_val > 0.0F) 
					err -= actual_val * __logf(max(predicted_val, 1.0e-20F));
			}

		#if __CUDA_ARCH__ < 300
			volatile float * arr = arr_sh;
			arr[thread_id] = err;
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			#else
				err += __shfl_down(err, tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			err = arr[thread_id];
			__syncthreads();
		#endif

			if (threadblock_size > 32)
			{
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = err;
				__syncthreads();
			}

			if (thread_id == 0)
			{
				for(int i = 1; i < (blockDim.x >> 5); ++i)
					err += arr_sh[i];
				double err_d = (double)err;

				atomicAdd(total_error, err_d);
			}
		}

		negative_log_likelihood_error_function_updater_cuda::negative_log_likelihood_error_function_updater_cuda()
		{
		}

		negative_log_likelihood_error_function_updater_cuda::~negative_log_likelihood_error_function_updater_cuda()
		{
		}

		const boost::uuids::uuid& negative_log_likelihood_error_function_updater_cuda::get_uuid() const
		{
			return negative_log_likelihood_error_function::function_guid;
		}

		void negative_log_likelihood_error_function_updater_cuda::enqueue_update_error_and_gradient(
			cudaStream_t stream_id,
			cuda_linear_buffer_device_smart_ptr gradient_buffer,
			cuda_linear_buffer_device_smart_ptr error_buffer,
			const_cuda_linear_buffer_device_smart_ptr actual_output_buffer,
			const_cuda_linear_buffer_device_smart_ptr predicted_output_buffer,
			unsigned int offset_entry_id,
			unsigned int neuron_count,
			unsigned int updater_entry_count) const
		{
			int threadblock_size = get_threadblock_size(neuron_count);
			int block_count = (neuron_count + threadblock_size - 1) / threadblock_size;
			dim3 grid_size(updater_entry_count, block_count, 1);
			dim3 block_size(threadblock_size, 1, 1);

			int smem_size = threadblock_size * sizeof(float);
			negative_log_likelihood_update_error_and_gradient_kernel<<<grid_size, block_size, smem_size, stream_id>>>(
				*gradient_buffer,
				*error_buffer,
				*actual_output_buffer,
				*predicted_output_buffer,
				offset_entry_id,
				neuron_count,
				updater_entry_count);
		}

		void negative_log_likelihood_error_function_updater_cuda::enqueue_update_error_and_gradient_fused_with_activation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device_smart_ptr gradient_buffer,
			cuda_linear_buffer_device_smart_ptr error_buffer,
			const_cuda_linear_buffer_device_smart_ptr actual_output_buffer,
			const_cuda_linear_buffer_device_smart_ptr predicted_output_buffer,
			unsigned int offset_entry_id,
			unsigned int neuron_count,
			unsigned int updater_entry_count) const
		{
			int threadblock_size = get_threadblock_size(neuron_count);
			dim3 grid_size(updater_entry_count, 1, 1);
			dim3 block_size(threadblock_size, 1, 1);

			int smem_size = threadblock_size * sizeof(float);
			negative_log_likelihood_update_error_and_gradient_fused_with_activation_kernel<<<grid_size, block_size, smem_size, stream_id>>>(
				*gradient_buffer,
				*error_buffer,
				*actual_output_buffer,
				*predicted_output_buffer,
				offset_entry_id,
				neuron_count,
				updater_entry_count);
		}

		int negative_log_likelihood_error_function_updater_cuda::get_threadblock_size(int output_neuron_count)
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

		const boost::uuids::uuid& negative_log_likelihood_error_function_updater_cuda::get_fusable_activation_uuid() const
		{
			return softmax_layer::layer_guid;
		}
	}
}
