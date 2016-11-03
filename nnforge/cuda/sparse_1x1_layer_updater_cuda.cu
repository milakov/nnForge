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

#include "sparse_1x1_layer_updater_cuda.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

#include "util_cuda.h"
#include "cudnn_util.h"
#include "neural_network_cusparse_exception.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cuda_exception.h"
#include "neural_network_cudnn_exception.h"
#include "../sparse_convolution_layer.h"
#include "int_fastdiv.h"

namespace nnforge
{
	namespace cuda
	{
		#define OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE 4
		__global__ void sparse_1x1_backprop_upd_kernel(
			const float * __restrict output_errors,
			float * __restrict input_errors,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_feature_map_count,
			int entry_count,
			int entry32_block_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_feature_map_count)
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
		template<bool single_entry_pass>
		__global__ void sparse_1x1_update_weights_kernel(
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			float * __restrict gradient_weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			int output_feature_map_count,
			int entry_count,
			int entry32_block_size)
		{
			int row_id = blockIdx.y * blockDim.y + threadIdx.y;
			if (row_id >= output_feature_map_count)
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

			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
				#pragma unroll
				for(int i = 0; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
					sums[i] += __shfl_xor(sums[i], tx);
			}
			#pragma unroll
			for(int i = 1; i < OUTPUT_ELEM_COUNT_BLOCK_SIZE; ++i)
				if (lane_id == i)
					sums[0] = sums[i];

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
/*
		template<int DIMENSION_COUNT, bool add_update_to_destination>
		__global__ void sparse_1x1_convert_to_strided(
			float * __restrict strided_nchw_values,
			const float * __restrict packed_cnhw_values,
			array_by_val<int_fastdiv, DIMENSION_COUNT> input_compacted_sizes,
			array_by_val<int, DIMENSION_COUNT> strides,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			int neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
			if ((neuron_id < neuron_count_per_feature_map) && (feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				int xyz[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / input_compacted_sizes[i];
					xyz[i] = remainder - input_compacted_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyz[DIMENSION_COUNT - 1] = remainder;

				int compacted_offset = (feature_map_id * entry_count + entry_id) * neuron_count_per_feature_map + neuron_id;

				int strided_offset = entry_id * feature_map_count + feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					strided_offset = strided_offset * input_sizes[i] + xyz[i] * strides[i];

				if (add_update_to_destination)
					strided_nchw_values[strided_offset] += packed_cnhw_values[compacted_offset];
				else
					strided_nchw_values[strided_offset] = packed_cnhw_values[compacted_offset];
			}
		}
*/
		sparse_1x1_layer_updater_cuda::sparse_1x1_layer_updater_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_strided_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_converted_NHWC_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_converted_CNHW_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		sparse_1x1_layer_updater_cuda::~sparse_1x1_layer_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(input_strided_data_desc);
			cudnnDestroyTensorDescriptor(input_converted_NHWC_data_desc);
			cudnnDestroyTensorDescriptor(input_converted_CNHW_data_desc);
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void sparse_1x1_layer_updater_cuda::enqueue_forward_propagation(
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
			// Convert input data strided NCHW to packed CNHW format
			if (unit_stride)
			{
				cuda_util::transpose23(
					*cuda_config,
					*input_buffers[0],
					*temporary_working_per_entry_buffer,
					input_elem_count_per_feature_map_list[0],
					input_configuration_specific_list[0].feature_map_count,
					entry_count,
					stream_id);
			}
			else
			{
				std::vector<unsigned int> input_converted_CNHW_strides = input_converted_CNHW_strides_base;
				input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 2] = input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 1] * entry_count;
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					input_strided_data_desc,
					input_strided_config,
					entry_count,
					input_strides);
				cudnn_util::set_tensor_descriptor(
					input_converted_CNHW_data_desc,
					input_strided_config,
					entry_count,
					input_converted_CNHW_strides);
				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnAddTensor(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_strided_data_desc,
					*input_buffers[0],
					&beta,
					input_converted_CNHW_data_desc,
					*temporary_working_per_entry_buffer));
			}

			{
				cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				cusparseMatDescr_t mat_descr;
				cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
				cusparse_safe_call(cusparseScsrmm2(
					cuda_config->get_cusparse_handle(),
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					CUSPARSE_OPERATION_TRANSPOSE,
					output_configuration_specific.feature_map_count,
					entry_count * output_elem_count_per_feature_map,
					input_strided_config.feature_map_count,
					feature_map_connection_count,
					&alpha,
					mat_descr,
					*data[0],
					*data_custom[1],
					*data_custom[0],
					*temporary_working_per_entry_buffer,
					entry_count * output_elem_count_per_feature_map,
					&beta,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					output_configuration_specific.feature_map_count));
			}

			// Convert output from NHWC to NCHW
			{
				cuda_util::transpose(
					*cuda_config,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					*output_buffer,
					output_configuration_specific.feature_map_count,
					output_elem_count_per_feature_map,
					entry_count,
					stream_id);
			}

			// Add bias
			if (bias)
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnAddTensor(
					cuda_config->get_cudnn_handle(),
					&alpha,
					bias_desc,
					*data[1],
					&beta,
					output_data_desc,
					*output_buffer));
			}
		}

		void sparse_1x1_layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			// Convert output errors from NCHW to CNHW
			{
				cuda_util::transpose23(
					*cuda_config,
					*output_errors_buffer,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);
			}

			{
				cuda_util::set_with_value(
					*cuda_config,
					*temporary_working_per_entry_buffer,
					0.0F,
					input_converted_elem_count_per_entry_aligned * entry_count,
					stream_id);

				std::pair<int, int> entry32_block_size_and_count = get_entry32_backprop_block_size_and_count(entry_count * output_elem_count_per_feature_map);
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					32 * ((max_column_index_count_per_row + OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BACKPROP_BLOCK_SIZE),
					output_configuration_specific.feature_map_count,
					entry32_block_size_and_count.second,
					32);
				sparse_1x1_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					*temporary_working_per_entry_buffer,
					*data[0],
					*data_custom[0],
					*data_custom[1],
					output_configuration_specific.feature_map_count,
					entry_count * output_elem_count_per_feature_map,
					entry32_block_size_and_count.first);
			}

			// Convert input errors from CNHW to NCHW
			{
				if ((!add_update_to_destination) && (!unit_stride))
				{
					cuda_util::set_with_value(
						*cuda_config,
						*input_errors_buffer,
						0.0F,
						input_elem_count_per_entry_list[0] * entry_count,
						stream_id);
				}

				if (unit_stride)
				{
					cuda_util::transpose23(
						*cuda_config,
						*temporary_working_per_entry_buffer,
						*input_errors_buffer,
						input_elem_count_per_feature_map_list[0],
						entry_count,
						input_configuration_specific_list[0].feature_map_count,
						stream_id,
						add_update_to_destination);
				}
				else
				{
					std::vector<unsigned int> input_converted_CNHW_strides = input_converted_CNHW_strides_base;
					input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 2] = input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 1] * entry_count;
					cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
					cudnn_util::set_tensor_descriptor(
						input_strided_data_desc,
						input_strided_config,
						entry_count,
						input_strides);
					cudnn_util::set_tensor_descriptor(
						input_converted_CNHW_data_desc,
						input_strided_config,
						entry_count,
						input_converted_CNHW_strides);

					float alpha = 1.0F;
					float beta = add_update_to_destination ? 1.0F : 0.0F;
					cudnn_safe_call(cudnnAddTensor(
						cuda_config->get_cudnn_handle(),
						&alpha,
						input_converted_CNHW_data_desc,
						*temporary_working_per_entry_buffer,
						&beta,
						input_strided_data_desc,
						*input_errors_buffer));
				}
			}
		}

		void sparse_1x1_layer_updater_cuda::enqueue_backward_weights_propagation(
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& gradient,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			// Convert input data to packed CNHW format
			if (unit_stride)
			{
				cuda_util::transpose(
					*cuda_config,
					*input_neurons_buffers[0],
					*temporary_working_per_entry_buffer,
					input_elem_count_per_feature_map_list[0],
					input_configuration_specific_list[0].feature_map_count,
					entry_count,
					stream_id);
			}
			else
			{
				std::vector<unsigned int> input_converted_CNHW_strides = input_converted_CNHW_strides_base;
				input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 2] = input_converted_CNHW_strides[input_converted_CNHW_strides.size() - 1] * entry_count;
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					input_strided_data_desc,
					input_strided_config,
					entry_count,
					input_strides);
				cudnn_util::set_tensor_descriptor(
					input_converted_CNHW_data_desc,
					input_strided_config,
					entry_count,
					input_converted_CNHW_strides);
				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnAddTensor(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_strided_data_desc,
					*input_neurons_buffers[0],
					&beta,
					input_converted_CNHW_data_desc,
					*temporary_working_per_entry_buffer));
			}

			// Convert output from NCHW to CNHW 
			{
				cuda_util::transpose23(
					*cuda_config,
					*output_errors_buffer,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);
			}

			{
				std::pair<int, int> entry32_block_size_and_count = get_entry32_update_block_size_and_count(output_elem_count_per_feature_map * entry_count);
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					32 * ((max_column_index_count_per_row + OUTPUT_ELEM_COUNT_BLOCK_SIZE - 1) / OUTPUT_ELEM_COUNT_BLOCK_SIZE),
					output_configuration_specific.feature_map_count,
					entry32_block_size_and_count.second,
					32);
				if (entry32_block_size_and_count.second > 1)
				{
					sparse_1x1_update_weights_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						((const float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
						*temporary_working_per_entry_buffer,
						*gradient[0],
						*data_custom[0],
						*data_custom[1],
						output_configuration_specific.feature_map_count,
						output_elem_count_per_feature_map * entry_count,
						entry32_block_size_and_count.first);
				}
				else
				{
					sparse_1x1_update_weights_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						((const float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
						*temporary_working_per_entry_buffer,
						*gradient[0],
						*data_custom[0],
						*data_custom[1],
						output_configuration_specific.feature_map_count,
						output_elem_count_per_feature_map * entry_count,
						entry32_block_size_and_count.first);
				}
			}

			// Update biases
			if (bias)
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnConvolutionBackwardBias(
					cuda_config->get_cudnn_handle(),
					&alpha,
					output_data_desc,
					*output_errors_buffer,
					&beta,
					bias_desc,
					*gradient[1]));
			}
		}

		void sparse_1x1_layer_updater_cuda::updater_configured()
		{
			std::shared_ptr<const sparse_convolution_layer> layer_derived = std::dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;
			bias = layer_derived->bias;
			unit_stride = (layer_derived->strides == std::vector<unsigned int>(layer_derived->strides.size(), 1));

			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));

			int input_data_single_update_32block_entry_size = input_configuration_specific_list[0].feature_map_count * 32 * sizeof(float);
			max_entry32_update_block_size = std::max(1, cuda_config->l2_cache_size / 2 / input_data_single_update_32block_entry_size);

			int input_data_single_backprop_32block_entry_size = input_configuration_specific_list[0].feature_map_count * 32 * sizeof(float);
			max_entry32_backprop_block_size = std::max(1, cuda_config->l2_cache_size / 2 / input_data_single_backprop_32block_entry_size);

			const std::vector<unsigned int>& strides = layer_derived->strides;
			input_strided_config.feature_map_count = input_configuration_specific_list[0].feature_map_count;
			input_strided_config.dimension_sizes = output_configuration_specific.dimension_sizes;
			input_strides.resize(strides.size() + 1);
			unsigned int dim_size = 1;
			for(int i = 0; i < strides.size(); ++i)
			{
				*(input_strides.begin() + i) = strides[i] * dim_size;
				dim_size *= input_configuration_specific_list[0].dimension_sizes[i];
			}
			input_strides[strides.size()] = dim_size;

			input_converted_NHWC_strides.resize(strides.size() + 2);
			input_converted_NHWC_strides[strides.size()] = 1;
			dim_size = input_strided_config.feature_map_count;
			for(int i = 0; i < strides.size(); ++i)
			{
				input_converted_NHWC_strides[i] = dim_size;
				dim_size *= output_configuration_specific.dimension_sizes[i];
			}
			input_converted_NHWC_strides.back() = dim_size;

			input_converted_CNHW_strides_base.resize(strides.size() + 2);
			dim_size = 1;
			for(int i = 0; i < strides.size(); ++i)
			{
				input_converted_CNHW_strides_base[i] = dim_size;
				dim_size *= output_configuration_specific.dimension_sizes[i];
			}
			input_converted_CNHW_strides_base[strides.size() + 1] = dim_size;

			input_converted_elem_count_per_entry_aligned = (input_strided_config.get_neuron_count() + 4 - 1) / 4 * 4;
			output_elem_count_per_entry_aligned = (output_configuration_specific.get_neuron_count() + 4 - 1) / 4 * 4;
		}

		void sparse_1x1_layer_updater_cuda::notify_data_custom(layer_data_custom::const_ptr host_data_custom)
		{
			max_column_index_count_per_row = 0;
			const std::vector<int>& row_indices = host_data_custom->at(1);
			for(int i = 0; i < row_indices.size() - 1; ++i)
				max_column_index_count_per_row = std::max(max_column_index_count_per_row, row_indices[i + 1] - row_indices[i]);
		}

		size_t sparse_1x1_layer_updater_cuda::get_temporary_working_per_entry_buffer_size(const layer_action& action) const
		{
			if ((action.get_action_type() == layer_action::forward) || (action.get_action_type() == layer_action::backward_data) || (action.get_action_type() == layer_action::backward_weights))
				return (input_converted_elem_count_per_entry_aligned * sizeof(float)) + (output_elem_count_per_entry_aligned * sizeof(float));
			else
				return layer_updater_cuda::get_temporary_working_per_entry_buffer_size(action);
		}

		std::pair<int, int> sparse_1x1_layer_updater_cuda::get_entry32_update_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = (entry_count + 32 - 1) / 32;

			if (candidate_block_size <= max_entry32_update_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_entry32_update_block_size - 1) / max_entry32_update_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}

		std::pair<int, int> sparse_1x1_layer_updater_cuda::get_entry32_backprop_block_size_and_count(unsigned int entry_count) const
		{
			int candidate_block_size = (entry_count + 32 - 1) / 32;

			if (candidate_block_size <= max_entry32_backprop_block_size)
				return std::make_pair(candidate_block_size, 1);

			int candidate_block_count2 = (candidate_block_size + max_entry32_backprop_block_size - 1) / max_entry32_backprop_block_size;
			int candidate_block_size2 = (candidate_block_size + candidate_block_count2 - 1) / candidate_block_count2;

			return std::make_pair(candidate_block_size2, candidate_block_count2);
		}

		bool sparse_1x1_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool sparse_1x1_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool sparse_1x1_layer_updater_cuda::is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			return true;
		}
	}
}
