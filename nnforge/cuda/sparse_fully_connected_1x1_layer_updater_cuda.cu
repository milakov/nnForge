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
		__global__ void sparse_fully_connected_1x1_update_biases_upd_kernel(
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

		#define OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE 4
		__global__ void sparse_fully_connected_1x1_backprop_upd_kernel(
			const float * __restrict output_errors,
			float * __restrict input_errors,
			const float * __restrict weights,
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
			int base_column_index_offset = (thread_id_x >> 5) * OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE;
			int base_nnz_index = start_column_index + base_column_index_offset;
			if (base_nnz_index >= end_column_index)
				return;

			int max_valid_lane = min(OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE, end_column_index - base_nnz_index);

			bool valid[OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE];
			int column_ids[OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE];
			float w[OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE; ++i)
			{
				valid[i] = (i < max_valid_lane);
				int index = valid[i] ? base_nnz_index + i : (end_column_index - 1);
				column_ids[i] = __load_nc(column_indices + index);
				w[i] = __load_nc(weights + index);
			}
			int base_entry_id = ((blockIdx.z * blockDim.z + threadIdx.z) << 5) * entry32_block_size;
			if (base_entry_id >= entry_count)
				return;

			int lane_id = thread_id_x & 31;
			int current_entry_id = base_entry_id + lane_id;
			const float * base_output_errors = output_errors + row_id * entry_count;
			for(int j = 0; j < entry32_block_size; ++j, current_entry_id += 32)
			{
				if (current_entry_id < entry_count)
				{
					float output_error = __load_nc(base_output_errors + current_entry_id);
					#pragma unroll
					for(int i = 0; i < OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE; ++i)
						if (valid[i])
							atomicAdd(input_errors + column_ids[i] * entry_count + current_entry_id, output_error * w[i]);
				}
			}
		}


		#define OUTPUT_ELEM_COUNT_BLOCK_SIZE 4
		extern __shared__ float arr_sh[];
		template<bool single_entry_pass>
		__global__ void sparse_fully_connected_1x1_update_weights_kernel(
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
			int base_column_index_offset = (thread_id_x >> 5) * OUTPUT_ELEM_COUNT_BLOCK_SIZE;
			int base_nnz_index = start_column_index + base_column_index_offset;
			if (base_nnz_index >= end_column_index)
				return;

			int max_valid_lane = min(OUTPUT_ELEM_COUNT_BLOCK_SIZE, end_column_index - base_nnz_index);
			bool valid[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				valid[i] = (i < max_valid_lane);
			int column_ids[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				column_ids[i] = __load_nc(column_indices + (valid[i] ? base_nnz_index + i : (end_column_index - 1)));
			int base_entry_id = ((blockIdx.z * blockDim.z + threadIdx.z) << 5) * entry32_block_size;
			if (base_entry_id >= entry_count)
				return;

			int lane_id = thread_id_x & 31;
			int current_entry_id = base_entry_id + lane_id;

			float sums[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				sums[i] = 0.0F;
			const float * base_output_errors = output_errors + row_id * entry_count;
			for(int j = 0; j < entry32_block_size; ++j, current_entry_id += 32)
			{
				if (current_entry_id < entry_count)
				{
					float output_error = __load_nc(base_output_errors + current_entry_id);
					#pragma unroll
					for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
						sums[i] += __load_nc(input_neurons + column_ids[i] * entry_count + current_entry_id) * output_error;
				}
			}

		#if __CUDA_ARCH__ < 300
			int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;
			int warp_id = thread_id >> 5;
			volatile float * arr = arr_sh;
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				arr[warp_id * (32 * OUTPUT_ELEM_COUNT_BLOCK_SIZE) + i * 32 + lane_id] = sums[i];
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
				{
					#pragma unroll
					for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
						arr[warp_id * (32 * OUTPUT_ELEM_COUNT_BLOCK_SIZE) + i * 32 + lane_id] += arr[warp_id * (32 * OUTPUT_ELEM_COUNT_BLOCK_SIZE) + i * 32 + lane_id + tx];
				}
			#else
				#pragma unroll
				for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
					sums[i] += __shfl_xor(sums[i], tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			if (lane_id < max_valid_lane)
				sums[0] = arr[warp_id * (32 * OUTPUT_ELEM_COUNT_BLOCK_SIZE) + lane_id * 32];
		#else
			#pragma unroll
			for(int i = 1; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				if (lane_id == i)
					sums[0] = sums[i];
		#endif

			if (lane_id < max_valid_lane)
			{
				if (single_entry_pass)
				{
					gradient_weights[base_nnz_index + lane_id] += sums[0];
				}
				else
				{
					atomicAdd(gradient_weights + base_nnz_index + lane_id, sums[0]);
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
			// Copy bias
			cuda_util::duplicate_vector(
				*cuda_config,
				*data[1],
				*output_neurons_buffer,
				output_elem_count_per_entry,
				entry_count,
				stream_id);

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
			// Too slow
			/*
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
				*/

			cuda_util::set_with_value(
				*cuda_config,
				*additional_buffers[0],
				0.0F,
				input_elem_count_per_entry * entry_count,
				stream_id);

			std::pair<int, int> entry32_block_size_and_count = get_entry32_backprop_block_size_and_count(entry_count);
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				32 * ((max_column_index_count_per_row + OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE),
				output_elem_count_per_entry,
				entry32_block_size_and_count.second,
				32);
			sparse_fully_connected_1x1_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*additional_buffers[1],
				*additional_buffers[0],
				*data[0],
				*data_custom[0],
				*data_custom[1],
				output_elem_count_per_entry,
				entry_count,
				entry32_block_size_and_count.first);

			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			// transpose input
			{
				float alpha = 1.0F;
				float beta = 0.0F;
				cublas_safe_call(cublasSgeam(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					input_elem_count_per_entry,
					entry_count,
					&alpha,
					*additional_buffers[0],
					entry_count,
					&beta,
					*input_errors_buffer,
					input_elem_count_per_entry,
					*input_errors_buffer,
					input_elem_count_per_entry));
			}
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

				std::pair<int, int> entry32_block_size_and_count = get_entry32_update_block_size_and_count(entry_count);
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					32 * ((max_column_index_count_per_row + OUTPUT_ELEM_COUNT_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BLOCK_SIZE),
					output_elem_count_per_entry,
					entry32_block_size_and_count.second,
					32);
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = (cuda_config->get_compute_capability() < 300) ? threadblock_size * OUTPUT_ELEM_COUNT_BLOCK_SIZE * sizeof(float) : 0;
				if (entry32_block_size_and_count.second > 1)
				{
					sparse_fully_connected_1x1_update_weights_kernel<false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
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
					sparse_fully_connected_1x1_update_weights_kernel<true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
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
				sparse_fully_connected_1x1_update_biases_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
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

			int input_data_single_update_32block_entry_size = input_elem_count_per_entry * 32 * sizeof(float);
			max_entry32_update_block_size = std::max(1, cuda_config->l2_cache_size / 2 / input_data_single_update_32block_entry_size);

			int input_data_single_backprop_32block_entry_size = input_elem_count_per_entry * 32 * sizeof(float);
			max_entry32_backprop_block_size = std::max(1, cuda_config->l2_cache_size / 2 / input_data_single_backprop_32block_entry_size);
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

		std::pair<int, int> sparse_fully_connected_1x1_layer_updater_cuda::get_entry32_update_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = (entry_count + 32 - 1) / 32;

			if (candidate_block_size <= max_entry32_update_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_entry32_update_block_size - 1) / max_entry32_update_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}

		std::pair<int, int> sparse_fully_connected_1x1_layer_updater_cuda::get_entry32_backprop_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = (entry_count + 32 - 1) / 32;

			if (candidate_block_size <= max_entry32_backprop_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_entry32_backprop_block_size - 1) / max_entry32_backprop_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}
	}
}
