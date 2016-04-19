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

#pragma once

#include "layer_tester_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "int_fastdiv.h"

#include "../max_subsampling_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		#define FEATURE_MAP_BLOCK_SIZE 4

		extern __shared__ float arr_sh[];

		template<int DIMENSION_COUNT,bool NONUNIT_WINDOW_X,bool IS_MIN, bool TRUNCATION>
		__global__ void max_subsampling_kernel(
			float * __restrict output,
			const float * __restrict input,
			array_by_val<int, DIMENSION_COUNT> subsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> strides,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> output_strides,
			int feature_map_subsampling_size,
			int entry_subsampling_size,
			int input_neuron_count_per_entry,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_entry_count,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int output_entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			int local_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
			int threadblock_size = blockDim.z * blockDim.y * blockDim.x;

			float * vals = arr_sh;

			bool in_bounds = (output_entry_id < output_entry_count) && (base_output_feature_map_id < output_feature_map_count) && (packed_config_id < packed_config_count);

			float res[FEATURE_MAP_BLOCK_SIZE];
			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			int window_x;
			int xyzw[DIMENSION_COUNT];
			int xyzw_input[DIMENSION_COUNT];
			if (in_bounds)
			{
				int remaining_part = packed_config_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				{
					xyzw[i] = remaining_part / output_strides[i];
					xyzw_input[i] = xyzw[i] * strides[i];
					remaining_part = remaining_part - output_strides[i] * xyzw[i];
				}
				window_x = remaining_part;

				int base_current_input_elem_id = output_entry_id * entry_subsampling_size * input_feature_map_count + base_output_feature_map_id * feature_map_subsampling_size;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					base_current_input_elem_id = base_current_input_elem_id * input_sizes[i] + xyzw_input[i];
				base_current_input_elem_id += window_x;

				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					res[i] = IS_MIN ? 1.0e37F : -1.0e37F;
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					item_valid[i - 1] = (base_output_feature_map_id + i < output_feature_map_count);

				bool fit_x = TRUNCATION ? (window_x < input_sizes[0] - xyzw_input[0]) : true;
				for(int en = 0; en < entry_subsampling_size; ++en)
				{
					int base_current_input_elem_id2 = base_current_input_elem_id;
					for(int fm = 0; fm < feature_map_subsampling_size; ++fm)
					{
						int current_input_elem_id = base_current_input_elem_id2;
						for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? subsampling_sizes[3] : 1); ++input_w)
						{
							bool fit_w = fit_x && (((DIMENSION_COUNT > 3) && TRUNCATION)? (input_w < input_sizes[3] - xyzw_input[3]) : true);
							for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? subsampling_sizes[2] : 1); ++input_z)
							{
								bool fit_z = fit_w && (((DIMENSION_COUNT > 2) && TRUNCATION) ? (input_z < input_sizes[2] - xyzw_input[2]) : true);
								for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? subsampling_sizes[1] : 1); ++input_y)
								{
									bool fit_y = fit_z && (((DIMENSION_COUNT > 1) && TRUNCATION) ? (input_y < input_sizes[1] - xyzw_input[1]) : true);
									if (fit_y)
									{
										float new_val[FEATURE_MAP_BLOCK_SIZE];
										new_val[0] = input[current_input_elem_id];
										#pragma unroll
										for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											if (item_valid[i - 1])
												new_val[i] = input[current_input_elem_id + input_neuron_count_per_feature_map * feature_map_subsampling_size * i];
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											res[i] = IS_MIN ? min(res[i], new_val[i]) : max(res[i], new_val[i]);
									}
									if (DIMENSION_COUNT > 1)
										current_input_elem_id += input_sizes[0];
								} // for input_y
								if (DIMENSION_COUNT > 2)
									current_input_elem_id += input_sizes[0] * (input_sizes[1] - subsampling_sizes[1]);
							} // for input_z
							if (DIMENSION_COUNT > 3)
								current_input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - subsampling_sizes[2]);
						} // for input_w
						base_current_input_elem_id2 += input_neuron_count_per_feature_map;
					} // for fm
					base_current_input_elem_id += input_neuron_count_per_entry;
				} // for en

				if (NONUNIT_WINDOW_X)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						vals[local_thread_id + threadblock_size * i] = res[i];
				}
			}

			if (NONUNIT_WINDOW_X)
				__syncthreads();

			if (in_bounds && (window_x == 0))
			{
				if (NONUNIT_WINDOW_X)
				{
					for(int j = 1; j < subsampling_sizes[0]; ++j)
					{
						local_thread_id++;
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						{
							float new_val = vals[local_thread_id + threadblock_size * i];
							res[i] = IS_MIN ? min(res[i], new_val) : max(res[i], new_val);
						}
					}
				}
				int offset = output_entry_id * output_feature_map_count + base_output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					offset = offset * output_sizes[i] + xyzw[i];
				output[offset] = res[0];
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					offset += output_neuron_count_per_feature_map;
					if (item_valid[i - 1])
						output[offset] = res[i];
				}
			}
		}

		template<int dimension_count>
		class max_subsampling_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			max_subsampling_layer_tester_cuda()
			{
			}

			virtual ~max_subsampling_layer_tester_cuda()
			{
			}

			virtual void enqueue_forward_propagation(
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
				int feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					subsampling_sizes[0]);

				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = (nonunit_window_x ? threadblock_size * sizeof(float) * FEATURE_MAP_BLOCK_SIZE : 0);

				if (is_min)
				{
					if (nonunit_window_x)
					{
						if (truncation)
							max_subsampling_kernel<dimension_count,true,true,true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
						else
							max_subsampling_kernel<dimension_count,true,true,false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
					}
					else
					{
						if (truncation)
							max_subsampling_kernel<dimension_count,false,true,true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
						else
							max_subsampling_kernel<dimension_count,false,true,false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
					}
				}
				else
				{
					if (nonunit_window_x)
					{
						if (truncation)
							max_subsampling_kernel<dimension_count,true,false,true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
						else
							max_subsampling_kernel<dimension_count,true,false,false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
					}
					else
					{
						if (truncation)
							max_subsampling_kernel<dimension_count,false,false,true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
						else
							max_subsampling_kernel<dimension_count,false,false,false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
								*output_buffer,
								*input_buffers[0],
								subsampling_sizes,
								strides,
								input_sizes,
								output_sizes,
								output_strides,
								feature_map_subsampling_size,
								entry_subsampling_size,
								input_elem_count_per_entry_list[0],
								input_elem_count_per_feature_map_list[0],
								output_elem_count_per_feature_map,
								input_configuration_specific_list[0].feature_map_count,
								output_configuration_specific.feature_map_count,
								entry_count,
								forward_packed_config_count);
					}
				}
			}

		protected:
			virtual void tester_configured()
			{
				nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

				is_min = layer_derived->is_min;

				feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
				entry_subsampling_size = layer_derived->entry_subsampling_size;

				std::vector<unsigned int> local_subsampling_sizes = layer_derived->subsampling_sizes;
				if (local_subsampling_sizes.empty())
					local_subsampling_sizes.push_back(1);
				std::vector<unsigned int> local_strides = layer_derived->strides;
				if (local_strides.empty())
					local_strides.push_back(1);
				std::vector<unsigned int> local_input_dimension_sizes = input_configuration_specific_list[0].dimension_sizes;
				if (local_input_dimension_sizes.empty())
					local_input_dimension_sizes.push_back(1);
				std::vector<unsigned int> local_output_dimension_sizes = output_configuration_specific.dimension_sizes;
				if (local_output_dimension_sizes.empty())
					local_output_dimension_sizes.push_back(1);

				int_fastdiv current_output_stride(local_subsampling_sizes[0]);
				truncation = false;
				for(int i = 0; i < dimension_count; ++i)
				{
					subsampling_sizes[i] = local_subsampling_sizes[i];
					strides[i] = local_strides[i];
					input_sizes[i] = local_input_dimension_sizes[i];
					output_sizes[i] = local_output_dimension_sizes[i];
					output_strides[i] = current_output_stride;
					current_output_stride = current_output_stride * output_sizes[i];
					truncation = truncation || (((output_sizes[i] - 1) * strides[i] + subsampling_sizes[i]) > input_sizes[i]);
				}

				forward_packed_config_count = subsampling_sizes[0];
				for(int i = 0; i < dimension_count; ++i)
					forward_packed_config_count *= output_sizes[i];

				nonunit_window_x = (subsampling_sizes[0] > 1);
			}

		private:
			bool is_min;
			int feature_map_subsampling_size;
			int entry_subsampling_size;
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> subsampling_sizes;
			array_by_val<int, dimension_count> strides;
			array_by_val<int_fastdiv, dimension_count> output_strides;
			unsigned int forward_packed_config_count;
			bool nonunit_window_x;
			bool truncation;
		};
	}
}
