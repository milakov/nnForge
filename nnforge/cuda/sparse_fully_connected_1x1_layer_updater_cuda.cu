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

#include "sparse_fully_connected_1x1_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cusparse_exception.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cuda_exception.h"
#include "../sparse_convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void copy_bias_sparse_upd_kernel(
			const float * __restrict biases,
			float * __restrict output,
			int output_neuron_count,
			int entry_count)
		{
			int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

			if ((output_neuron_id < output_neuron_count) && (entry_id < entry_count))
			{
				float bias = biases[output_neuron_id];
				float * current_output = output + (int)(entry_id * output_neuron_count + output_neuron_id);
				#pragma unroll
				for(int i = 0; i < 4; ++i)
				{
					if (entry_id < entry_count)
						*current_output = bias;
					current_output += output_neuron_count;
					entry_id++;
				}
			}
		}

		__global__ void sparse_fully_connected_update_biases_upd_kernel(
			float * __restrict gradient_biases,
			const float * __restrict output_errors,
			int block_size,
			int output_elem_count_per_entry,
			int entry_count,
			int block_count)
		{
			int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int block_id = blockIdx.y * blockDim.y + threadIdx.y;
			if ((output_neuron_id < output_elem_count_per_entry) && (block_id < block_count))
			{
				int base_entry_id = block_size * block_id;
				int iteration_count = min(entry_count - base_entry_id, block_size);
				const float * current_error = output_errors + (base_entry_id * output_elem_count_per_entry + output_neuron_id);
				float sum = 0.0F;
				for(int i = 0; i < iteration_count; ++i)
				{
					sum += *current_error;
					current_error += output_elem_count_per_entry;
				}
				atomicAdd(gradient_biases + output_neuron_id, sum);
			}
		}

		extern __shared__ float arr_sh[];
		template<bool single_entry_pass>
		__global__ void sparse_fully_connected_update_weights(
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			float * __restrict gradient_weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_elem_count_per_entry,
			int entry_count,
			int entry32_block_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_elem_count_per_entry)
				return;
			int start_column_index = __load_nc(row_ptrs + row_id);
			int end_column_index = __load_nc(row_ptrs + row_id + 1);
			int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
			int column_index_offset = thread_id_x >> 5;
			int nnz_index = start_column_index + column_index_offset;
			if (nnz_index >= end_column_index)
				return;
			int column_id = __load_nc(column_indices + nnz_index);
			int base_entry_id = ((blockIdx.z * blockDim.z + threadIdx.z) << 5) * entry32_block_size;
			if (base_entry_id >= entry_count)
				return;

			int lane_id = thread_id_x & 31;
			int current_entry_id = base_entry_id + lane_id;

			float sum = 0.0F;
			const float * base_input_neurons = input_neurons + column_id * entry_count;
			const float * base_output_errors = output_errors + row_id * entry_count;
			for(int i = 0; i < entry32_block_size; ++i, current_entry_id += 32)
			{
				if (current_entry_id < entry_count)
					sum += __load_nc(base_input_neurons + current_entry_id) * __load_nc(base_output_errors + current_entry_id);
			}

		#if __CUDA_ARCH__ < 300
			int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;
			volatile float * arr = arr_sh;
			arr[thread_id] = sum;
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			#else
				sum += __shfl_down(sum, tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			sum = arr[thread_id];
		#endif

			if (lane_id == 0)
			{
				if (single_entry_pass)
				{
					gradient_weights[nnz_index] += sum;
				}
				else
				{
					atomicAdd(gradient_weights + nnz_index, sum);
				}
			}
		}

		sparse_fully_connected_1x1_layer_updater_cuda::sparse_fully_connected_1x1_layer_updater_cuda()
		{
		}

		sparse_fully_connected_1x1_layer_updater_cuda::~sparse_fully_connected_1x1_layer_updater_cuda()
		{
		}

		void sparse_fully_connected_1x1_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_entry,
				(entry_count + 4 - 1) / 4,
				1);
			copy_bias_sparse_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*data[1],
				*output_neurons_buffer,
				output_elem_count_per_entry,
				entry_count);

			cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 1.0F;
			cusparseMatDescr_t mat_descr;
			cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
			cusparse_safe_call(cusparseScsrmm(
				cuda_config->get_cusparse_handle(),
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				output_elem_count_per_entry,
				entry_count,
				input_elem_count_per_entry,
				feature_map_connection_count,
				&alpha,
				mat_descr,
				*data[0],
				*data_custom[1],
				*data_custom[0],
				(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
				input_elem_count_per_entry,
				&beta,
				*output_neurons_buffer,
				output_elem_count_per_entry));
		}

		void sparse_fully_connected_1x1_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 0.0F;
			cusparseMatDescr_t mat_descr;
			cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
			cusparse_safe_call(cusparseScsrmm(
				cuda_config->get_cusparse_handle(),
				CUSPARSE_OPERATION_TRANSPOSE,
				output_elem_count_per_entry,
				entry_count,
				input_elem_count_per_entry,
				feature_map_connection_count,
				&alpha,
				mat_descr,
				*data[0],
				*data_custom[1],
				*data_custom[0],
				*output_errors_buffer,
				output_elem_count_per_entry,
				&beta,
				*input_errors_buffer,
				input_elem_count_per_entry));
		}

		void sparse_fully_connected_1x1_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			// Update weights
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				// transpose input
				{
					float alpha = 1.0F;
					float beta = 0.0F;
					cublas_safe_call(cublasSgeam(
						cuda_config->get_cublas_handle(),
						CUBLAS_OP_T,
						CUBLAS_OP_N,
						entry_count,
						input_elem_count_per_entry,
						&alpha,
						(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
						input_elem_count_per_entry,
						&beta,
						*additional_buffers[0],
						entry_count,
						*additional_buffers[0],
						entry_count));
				}
				// transpose output
				{
					float alpha = 1.0F;
					float beta = 0.0F;
					cublas_safe_call(cublasSgeam(
						cuda_config->get_cublas_handle(),
						CUBLAS_OP_T,
						CUBLAS_OP_N,
						entry_count,
						output_elem_count_per_entry,
						&alpha,
						*output_errors_buffer,
						output_elem_count_per_entry,
						&beta,
						*additional_buffers[1],
						entry_count,
						*additional_buffers[1],
						entry_count));
				}

				std::pair<int, int> entry32_block_size_and_count = get_entry32_block_size_and_count(entry_count);
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					32 * max_column_index_count_per_row,
					output_elem_count_per_entry,
					entry32_block_size_and_count.second,
					32);
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				if (entry32_block_size_and_count.second > 1)
				{
					sparse_fully_connected_update_weights<false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*additional_buffers[1],
						*additional_buffers[0],
						*gradient[0],
						*data_custom[0],
						*data_custom[1],
						output_elem_count_per_entry,
						entry_count,
						entry32_block_size_and_count.first);
				}
				else
				{
					sparse_fully_connected_update_weights<true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*additional_buffers[1],
						*additional_buffers[0],
						*gradient[0],
						*data_custom[0],
						*data_custom[1],
						output_elem_count_per_entry,
						entry_count,
						entry32_block_size_and_count.first);
				}
			}

			// Update biases
			{
				int block_size = get_block_size(entry_count);
				int block_count = (entry_count + block_size - 1) / block_size;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					output_elem_count_per_entry,
					block_count,
					1);
				sparse_fully_connected_update_biases_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*gradient[1],
					*output_errors_buffer,
					block_size,
					output_elem_count_per_entry,
					entry_count,
					block_count);
			}
		}

		bool sparse_fully_connected_1x1_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		int sparse_fully_connected_1x1_layer_updater_cuda::get_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}

		void sparse_fully_connected_1x1_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;

			int input_data_single_32block_entry_size = (input_elem_count_per_entry + output_elem_count_per_entry) * 32 * sizeof(float);
			max_entry32_block_size = std::max(1, cuda_config->l2_cache_size / 2 / input_data_single_32block_entry_size);
		}

		std::vector<size_t> sparse_fully_connected_1x1_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_entry * sizeof(float));
			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		void sparse_fully_connected_1x1_layer_updater_cuda::notify_data_custom(const_layer_data_custom_smart_ptr host_data_custom)
		{
			max_column_index_count_per_row = 0;
			const std::vector<int>& row_indices = host_data_custom->at(1);
			for(int i = 0; i < row_indices.size() - 1; ++i)
				max_column_index_count_per_row = std::max(max_column_index_count_per_row, row_indices[i + 1] - row_indices[i]);
		}

		std::pair<int, int> sparse_fully_connected_1x1_layer_updater_cuda::get_entry32_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = (entry_count + 32 - 1) / 32;

			if (candidate_block_size <= max_entry32_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_entry32_block_size - 1) / max_entry32_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}
	}
}
