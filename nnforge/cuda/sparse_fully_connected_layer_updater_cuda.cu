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

#include "sparse_fully_connected_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "../sparse_convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		extern __shared__ float arr_sh[];

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

		#define OUTPUT_ELEM_COUNT_BLOCK_SIZE 4
		__global__ void sparse_fully_connected_upd_kernel(
			float * __restrict output_neurons,
			const float * __restrict input_neurons,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_elem_count_per_entry,
			int input_elem_count_per_entry,
			int entry_count,
			int input_feature_map_block_size,
			int window_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_elem_count_per_entry)
				return;
			int start_column_index = __load_nc(row_ptrs + row_id);
			int end_column_index = __load_nc(row_ptrs + row_id + 1);
			int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
			int base_column_index_offset = (thread_id_x >> 5) * input_feature_map_block_size;
			int base_nnz_index = start_column_index + base_column_index_offset;
			if (base_nnz_index >= end_column_index)
				return;

			int base_entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * OUTPUT_ELEM_COUNT_BLOCK_SIZE;
			if (base_entry_id >= entry_count)
				return;

			int lane_id = thread_id_x & 31;
			int it_count = min(input_feature_map_block_size, end_column_index - base_nnz_index);

			int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;
			int warp_id = thread_id >> 5;

			volatile int * column_indices_sh = (int *)arr_sh;
			if (lane_id < it_count)
				column_indices_sh[thread_id] = column_indices[base_nnz_index + lane_id];

			int window_it_count = (window_size + 31) >> 5;

			bool valid[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			int entry_ids[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
			{
				valid[i] = (i < (entry_count - base_entry_id));
				entry_ids[i] = valid[i] ? (base_entry_id + i) : (entry_count - 1);
			}

			float sums[OUTPUT_ELEM_COUNT_BLOCK_SIZE];
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				sums[i] = 0.0F;

			for(int i = 0; i < it_count; ++i)
			{
				int index = base_nnz_index + i;
				int column_id = column_indices_sh[warp_id * 32 + i];
				int local_weight_id = lane_id;
				for(int j = 0; j < window_it_count; ++j)
				{
					if (local_weight_id < window_size)
					{
						float w = __load_nc(weights + (int)(index * window_size + local_weight_id));
						#pragma unroll
						for(int k = 0; k < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++k)
						{
							float inp = __load_nc(input_neurons + entry_ids[k] * input_elem_count_per_entry + column_id * window_size + local_weight_id);
							sums[k] += w * inp;
						}
					}
					local_weight_id += 32;
				}
			}

		#if __CUDA_ARCH__ < 300
			int threadblock_size = blockDim.x * blockDim.y * blockDim.z;
			volatile float * arr = arr_sh;
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				arr[i * threadblock_size + thread_id] = sums[i];
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
				{
					#pragma unroll
					for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
						arr[i * threadblock_size + thread_id] += arr[i * threadblock_size + thread_id + tx];
				}
			#else
				#pragma unroll
				for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
					sums[i] += __shfl_xor(sums[i], tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				sums[i] = arr[i * threadblock_size + thread_id];
		#endif

			if (lane_id == 0)
			{
				for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
					if (valid[i])
						atomicAdd(output_neurons + (base_entry_id + i) * output_elem_count_per_entry + row_id, sums[i]);
			}
		}

		#define OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE 4
		__global__ void sparse_fully_connected_backprop_upd_kernel(
			const float * __restrict output_errors,
			float * __restrict input_errors,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_elem_count_per_entry,
			int input_elem_count_per_entry,
			int entry_count,
			int window_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_elem_count_per_entry)
				return;
			int start_column_index = __load_nc(row_ptrs + row_id);
			int end_column_index = __load_nc(row_ptrs + row_id + 1);
			int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
			int base_column_index_offset = thread_id_x >> 5;
			int base_nnz_index = start_column_index + base_column_index_offset;
			if (base_nnz_index >= end_column_index)
				return;
			int base_entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE;
			if (base_entry_id >= entry_count)
				return;

			bool valid[OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE];
			int entry_ids[OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE];
			int max_local_entry_count = min(OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE, entry_count - base_entry_id);
			#pragma unroll
			for(int i = 0; i < OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE; ++i)
			{
				valid[i] = (i < max_local_entry_count);
				entry_ids[i] = valid[i] ? (base_entry_id + i) : (entry_count - 1);
			}

			int column_id = __load_nc(column_indices + base_nnz_index);

			int window_it_count = (window_size + 31) >> 5;

			int lane_id = thread_id_x & 31;
			int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;
			int warp_id = thread_id >> 5;
			volatile float * output_errors_sh = arr_sh + warp_id * OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE;
			if (lane_id < max_local_entry_count)
					output_errors_sh[lane_id] = __load_nc(output_errors + (int)((base_entry_id + lane_id) * output_elem_count_per_entry + row_id));

			int local_weight_id = lane_id;
			for(int j = 0; j < window_it_count; ++j)
			{
				if (local_weight_id < window_size)
				{
					float w = __load_nc(weights + (int)(base_nnz_index * window_size + local_weight_id));
					#pragma unroll
					for(int k = 0; k < OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE; ++k)
					{
						if (valid[k])
						{
							float input_error = output_errors_sh[k] * w;
							atomicAdd(input_errors + (int)(entry_ids[k] * input_elem_count_per_entry + column_id * window_size + local_weight_id), input_error);
						}
					}
				}
				local_weight_id += 32;
			}
		}
		
		__global__ void sparse_fully_connected_update_weights_kernel(
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			float * __restrict gradient_weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_elem_count_per_entry,
			int input_elem_count_per_entry,
			int entry_block_size,
			int entry_count,
			int window_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_elem_count_per_entry)
				return;
			int start_column_index = __load_nc(row_ptrs + row_id);
			int end_column_index = __load_nc(row_ptrs + row_id + 1);
			int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
			int base_column_index_offset = thread_id_x >> 5;
			int base_nnz_index = start_column_index + base_column_index_offset;
			if (base_nnz_index >= end_column_index)
				return;
			int base_entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * entry_block_size;
			if (base_entry_id >= entry_count)
				return;

			int local_entry_count = min(entry_block_size, entry_count - base_entry_id);

			int column_id = __load_nc(column_indices + base_nnz_index);

			int window_it_count = (window_size + 31) >> 5;

			int lane_id = thread_id_x & 31;
			int thread_id = blockDim.x * (threadIdx.z * blockDim.y + threadIdx.y) + threadIdx.x;
			int warp_id = thread_id >> 5;
			volatile float * output_errors_sh = arr_sh + warp_id * entry_block_size;
			if (lane_id < local_entry_count)
					output_errors_sh[lane_id] = __load_nc(output_errors + (int)((base_entry_id + lane_id) * output_elem_count_per_entry + row_id));

			int local_weight_id = lane_id;
			for(int j = 0; j < window_it_count; ++j)
			{
				if (local_weight_id < window_size)
				{
					float sum = 0.0F;
					const float * current_input_neurons = input_neurons + base_entry_id * input_elem_count_per_entry + column_id * window_size + local_weight_id;
					for(int k = 0; k < local_entry_count; ++k)
					{
						sum += output_errors_sh[k] * __load_nc(current_input_neurons);
						current_input_neurons += input_elem_count_per_entry;
					}
					atomicAdd(gradient_weights + (int)(base_nnz_index * window_size + local_weight_id), sum);
				}
				local_weight_id += 32;
			}
		}
		
		const int sparse_fully_connected_layer_updater_cuda::max_input_feature_map_block_size = 32;
		const int sparse_fully_connected_layer_updater_cuda::absolute_min_update_entry_count_block_size = 4;
		const int sparse_fully_connected_layer_updater_cuda::absolute_max_update_entry_count_block_size = 32;

		sparse_fully_connected_layer_updater_cuda::sparse_fully_connected_layer_updater_cuda()
		{
		}

		sparse_fully_connected_layer_updater_cuda::~sparse_fully_connected_layer_updater_cuda()
		{
		}

		void sparse_fully_connected_layer_updater_cuda::enqueue_test(
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

			std::pair<int, int> input_feature_map_block_size_and_count = get_input_feature_map_block_size_and_count();
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				32 * input_feature_map_block_size_and_count.second,
				output_elem_count_per_entry,
				(entry_count + OUTPUT_ELEM_COUNT_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BLOCK_SIZE,
				32);
			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = (cuda_config->get_compute_capability() < 300) ? OUTPUT_ELEM_COUNT_BLOCK_SIZE * threadblock_size * sizeof(float) : threadblock_size * sizeof(float);
			sparse_fully_connected_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
				*output_neurons_buffer,
				*input_neurons_buffer,
				*data[0],
				*data_custom[0],
				*data_custom[1],
				output_elem_count_per_entry,
				input_elem_count_per_entry,
				entry_count,
				input_feature_map_block_size_and_count.first,
				window_size);
		}

		void sparse_fully_connected_layer_updater_cuda::enqueue_backprop(
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
			cuda_util::set_with_value(
				*cuda_config,
				*input_errors_buffer,
				0.0F,
				input_elem_count_per_entry * entry_count,
				stream_id);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				32 * max_column_index_count_per_row,
				output_elem_count_per_entry,
				(entry_count + OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE,
				32);
			int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
			int smem_size = (threadblock_size / 32) * OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE * sizeof(float);
			sparse_fully_connected_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
				*output_errors_buffer,
				*input_errors_buffer,
				*data[0],
				*data_custom[0],
				*data_custom[1],
				output_elem_count_per_entry,
				input_elem_count_per_entry,
				entry_count,
				window_size);
		}

		void sparse_fully_connected_layer_updater_cuda::enqueue_update_weights(
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
				std::pair<int, int> entry_block_size_and_count = get_update_entry_block_size_and_count(entry_count);
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					32 * max_column_index_count_per_row,
					output_elem_count_per_entry,
					entry_block_size_and_count.second,
					32);
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = (threadblock_size / 32) * entry_block_size_and_count.first * sizeof(float);
				sparse_fully_connected_update_weights_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*output_errors_buffer,
					*input_neurons_buffer,
					*gradient[0],
					*data_custom[0],
					*data_custom[1],
					output_elem_count_per_entry,
					input_elem_count_per_entry,
					entry_block_size_and_count.first,
					entry_count,
					window_size);
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

		bool sparse_fully_connected_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		int sparse_fully_connected_layer_updater_cuda::get_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}

		void sparse_fully_connected_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;

			window_size = 1;
			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_size *= *it;

			int input_data_single_backprop_entry_size = input_elem_count_per_entry * sizeof(float);
			max_update_entry_count_block_size = std::min(std::max(absolute_min_update_entry_count_block_size, cuda_config->l2_cache_size / 2 / input_data_single_backprop_entry_size), absolute_max_update_entry_count_block_size);
		}

		std::vector<size_t> sparse_fully_connected_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			return res;
		}

		void sparse_fully_connected_layer_updater_cuda::notify_data_custom(const_layer_data_custom_smart_ptr host_data_custom)
		{
			max_column_index_count_per_row = 0;
			const std::vector<int>& row_indices = host_data_custom->at(1);
			for(int i = 0; i < row_indices.size() - 1; ++i)
				max_column_index_count_per_row = std::max(max_column_index_count_per_row, row_indices[i + 1] - row_indices[i]);
		}

		std::pair<int, int> sparse_fully_connected_layer_updater_cuda::get_input_feature_map_block_size_and_count() const
		{
			int candidate_block_size = max_column_index_count_per_row;

			if (candidate_block_size <= max_input_feature_map_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_input_feature_map_block_size - 1) / max_input_feature_map_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}

		std::pair<int, int> sparse_fully_connected_layer_updater_cuda::get_update_entry_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = entry_count;

			if (candidate_block_size <= max_update_entry_count_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_update_entry_count_block_size - 1) / max_update_entry_count_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}
	}
}
