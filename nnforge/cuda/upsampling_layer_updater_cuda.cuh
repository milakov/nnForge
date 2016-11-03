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

#include "layer_updater_cuda.h"

#include <cuda_runtime.h>
#include <memory>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "int_fastdiv.h"

#include "../upsampling_layer.h"

namespace nnforge
{
	namespace cuda
	{
		#define FEATURE_MAP_BLOCK_SIZE 4

		extern __shared__ float arr_sh[];

		template<int DIMENSION_COUNT>
		__global__ void upsampling_upd_kernel(
			float * __restrict output,
			const float * __restrict input,
			array_by_val<int, DIMENSION_COUNT> upsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> strides,
			int feature_map_upsampling_size,
			int entry_upsampling_size,
			int output_neuron_count_per_entry,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_entry_count,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_input_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int input_entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (input_entry_id < input_entry_count) && (base_input_feature_map_id < input_feature_map_count) && (packed_config_id < packed_config_count);
			if (!in_bounds)
				return;

			int window_x;
			int xyzw[DIMENSION_COUNT];
			int remaining_part = packed_config_id;
			#pragma unroll
			for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
			{
				xyzw[i] = remaining_part / strides[i];
				remaining_part = remaining_part - strides[i] * xyzw[i];
			}
			window_x = remaining_part;

			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			#pragma unroll
			for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				item_valid[i - 1] = (base_input_feature_map_id + i < input_feature_map_count);

			float input_vals[FEATURE_MAP_BLOCK_SIZE];
			int input_offset = input_entry_id * input_feature_map_count + base_input_feature_map_id;
			#pragma unroll
			for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				input_offset = input_offset * input_sizes[i] + xyzw[i];

			input_vals[0] = input[input_offset];
			#pragma unroll
			for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
			{
				input_offset += input_neuron_count_per_feature_map;
				if (item_valid[i - 1])
					input_vals[i] = input[input_offset];
			}

			int base_current_output_elem_id = input_entry_id * entry_upsampling_size * output_feature_map_count + base_input_feature_map_id * feature_map_upsampling_size;
			#pragma unroll
			for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				base_current_output_elem_id = base_current_output_elem_id * input_sizes[i] + xyzw[i] * upsampling_sizes[i];
			base_current_output_elem_id += window_x;

			for(int en = 0; en < entry_upsampling_size; ++en)
			{
				int base_current_output_elem_id2 = base_current_output_elem_id;
				for(int fm = 0; fm < feature_map_upsampling_size; ++fm)
				{
					int current_output_elem_id = base_current_output_elem_id2;
					for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? upsampling_sizes[3] : 1); ++input_w)
					{
						for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? upsampling_sizes[2] : 1); ++input_z)
						{
							for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? upsampling_sizes[1] : 1); ++input_y)
							{
								output[current_output_elem_id] = input_vals[0];
								#pragma unroll
								for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									if (item_valid[i - 1])
										output[current_output_elem_id + output_neuron_count_per_feature_map * feature_map_upsampling_size * i] = input_vals[i];
								if (DIMENSION_COUNT > 1)
									current_output_elem_id += output_sizes[0];
							} // for input_y
							current_output_elem_id += output_sizes[0] * (output_sizes[1] - upsampling_sizes[1]);
						} // for input_z
						current_output_elem_id += output_sizes[1] * output_sizes[0] * (output_sizes[2] - upsampling_sizes[2]);
					} // for input_w
					base_current_output_elem_id2 += output_neuron_count_per_feature_map;
				} // for fm
				base_current_output_elem_id += output_neuron_count_per_entry;
			} // for en
		}

		template<int DIMENSION_COUNT,bool NONUNIT_WINDOW_X>
		__global__ void upsampling_backprop_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
			array_by_val<int, DIMENSION_COUNT> upsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> strides,
			int feature_map_upsampling_size,
			int entry_upsampling_size,
			int output_neuron_count_per_entry,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_entry_count,
			int packed_config_count,
			bool add_update_to_destination)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_input_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int input_entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			int local_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
			int threadblock_size = blockDim.z * blockDim.y * blockDim.x;

			float * vals = arr_sh;

			bool in_bounds = (input_entry_id < input_entry_count) && (base_input_feature_map_id < input_feature_map_count) && (packed_config_id < packed_config_count);

			float res[FEATURE_MAP_BLOCK_SIZE];
			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			int window_x;
			int xyzw[DIMENSION_COUNT];
			if (in_bounds)
			{
				int remaining_part = packed_config_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				{
					xyzw[i] = remaining_part / strides[i];
					remaining_part = remaining_part - strides[i] * xyzw[i];
				}
				window_x = remaining_part;

				int base_current_output_elem_id = input_entry_id * entry_upsampling_size * output_feature_map_count + base_input_feature_map_id * feature_map_upsampling_size;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					base_current_output_elem_id = base_current_output_elem_id * input_sizes[i] + xyzw[i] * upsampling_sizes[i];
				base_current_output_elem_id += window_x;

				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					res[i] = 0.0F;
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					item_valid[i - 1] = (base_input_feature_map_id + i < input_feature_map_count);

				for(int en = 0; en < entry_upsampling_size; ++en)
				{
					int base_current_output_elem_id2 = base_current_output_elem_id;
					for(int fm = 0; fm < feature_map_upsampling_size; ++fm)
					{
						int current_output_elem_id = base_current_output_elem_id2;
						for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? upsampling_sizes[3] : 1); ++input_w)
						{
							for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? upsampling_sizes[2] : 1); ++input_z)
							{
								for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? upsampling_sizes[1] : 1); ++input_y)
								{
									res[0] += output_errors[current_output_elem_id];
									#pragma unroll
									for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										if (item_valid[i - 1])
											res[i] += output_errors[current_output_elem_id + output_neuron_count_per_feature_map * feature_map_upsampling_size * i];
									if (DIMENSION_COUNT > 1)
										current_output_elem_id += output_sizes[0];
								} // for input_y
								if (DIMENSION_COUNT > 2)
									current_output_elem_id += output_sizes[0] * (output_sizes[1] - upsampling_sizes[1]);
							} // for input_z
							if (DIMENSION_COUNT > 3)
								current_output_elem_id += output_sizes[1] * output_sizes[0] * (output_sizes[2] - upsampling_sizes[2]);
						} // for input_w
						base_current_output_elem_id2 += output_neuron_count_per_feature_map;
					} // for fm
					base_current_output_elem_id += output_neuron_count_per_entry;
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
					for(int j = 1; j < upsampling_sizes[0]; ++j)
					{
						local_thread_id++;
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						{
							float new_val = vals[local_thread_id + threadblock_size * i];
							res[i] += new_val;
						}
					}
				}
				int offset = input_entry_id * input_feature_map_count + base_input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					offset = offset * input_sizes[i] + xyzw[i];

				if (add_update_to_destination)
				{
					float dst[FEATURE_MAP_BLOCK_SIZE];
					dst[0] = input_errors[offset];
					#pragma unroll
					for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						if (item_valid[i - 1])
							dst[i] = input_errors[offset + i * input_neuron_count_per_feature_map];
					input_errors[offset] = dst[0] + res[0];
					#pragma unroll
					for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						if (item_valid[i - 1])
							input_errors[offset + i * input_neuron_count_per_feature_map] = dst[i] + res[i];
				}
				else
				{
					input_errors[offset] = res[0];
					#pragma unroll
					for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						if (item_valid[i - 1])
							input_errors[offset + i * input_neuron_count_per_feature_map] = res[i];
				}
			}
		}

		template<int dimension_count>
		class upsampling_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			upsampling_layer_updater_cuda()
			{
			}

			virtual ~upsampling_layer_updater_cuda()
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
				cuda_linear_buffer_device::ptr temporary_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
				unsigned int entry_count)
			{
				int feature_map_block_count = (input_configuration_specific_list[0].feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					upsampling_sizes[0]);

				upsampling_upd_kernel<dimension_count><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					upsampling_sizes,
					input_sizes,
					output_sizes,
					strides,
					feature_map_upsampling_size,
					entry_upsampling_size,
					output_elem_count_per_entry,
					input_elem_count_per_feature_map_list[0],
					output_elem_count_per_feature_map,
					input_configuration_specific_list[0].feature_map_count,
					output_configuration_specific.feature_map_count,
					entry_count / entry_upsampling_size,
					forward_packed_config_count);
			}

			virtual void enqueue_backward_data_propagation(
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
				int feature_map_block_count = (input_configuration_specific_list[0].feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					upsampling_sizes[0]);

				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = (nonunit_window_x ? threadblock_size * sizeof(float) * FEATURE_MAP_BLOCK_SIZE : 0);

				if (nonunit_window_x)
					upsampling_backprop_upd_kernel<dimension_count,true><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						upsampling_sizes,
						input_sizes,
						output_sizes,
						strides,
						feature_map_upsampling_size,
						entry_upsampling_size,
						output_elem_count_per_entry,
						input_elem_count_per_feature_map_list[0],
						output_elem_count_per_feature_map,
						input_configuration_specific_list[0].feature_map_count,
						output_configuration_specific.feature_map_count,
						entry_count / entry_upsampling_size,
						forward_packed_config_count,
						add_update_to_destination);
				else
					upsampling_backprop_upd_kernel<dimension_count,false><<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						upsampling_sizes,
						input_sizes,
						output_sizes,
						strides,
						feature_map_upsampling_size,
						entry_upsampling_size,
						output_elem_count_per_entry,
						input_elem_count_per_feature_map_list[0],
						output_elem_count_per_feature_map,
						input_configuration_specific_list[0].feature_map_count,
						output_configuration_specific.feature_map_count,
						entry_count / entry_upsampling_size,
						forward_packed_config_count,
						add_update_to_destination);
			}

		protected:
			virtual void updater_configured()
			{
				std::shared_ptr<const upsampling_layer> layer_derived = std::dynamic_pointer_cast<const upsampling_layer>(layer_schema);

				feature_map_upsampling_size = layer_derived->feature_map_upsampling_size;
				entry_upsampling_size = layer_derived->entry_upsampling_size;

				std::vector<unsigned int> local_upsampling_sizes = layer_derived->upsampling_sizes;
				if (local_upsampling_sizes.empty())
					local_upsampling_sizes.push_back(1);
				std::vector<unsigned int> local_input_dimension_sizes = input_configuration_specific_list[0].dimension_sizes;
				if (local_input_dimension_sizes.empty())
					local_input_dimension_sizes.push_back(1);
				std::vector<unsigned int> local_output_dimension_sizes = output_configuration_specific.dimension_sizes;
				if (local_output_dimension_sizes.empty())
					local_output_dimension_sizes.push_back(1);

				int_fastdiv current_stride(local_upsampling_sizes[0]);
				for(int i = 0; i < dimension_count; ++i)
				{
					upsampling_sizes[i] = local_upsampling_sizes[i];
					input_sizes[i] = local_input_dimension_sizes[i];
					output_sizes[i] = local_output_dimension_sizes[i];
					strides[i] = current_stride;

					current_stride = current_stride * static_cast<int>(local_input_dimension_sizes[i]);
				}

				forward_packed_config_count = upsampling_sizes[0];
				for(int i = 0; i < dimension_count; ++i)
					forward_packed_config_count *= input_sizes[i];

				nonunit_window_x = (local_upsampling_sizes[0] > 1);
			}

			bool is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
			{
				return false;
			}

			bool is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
			{
				return false;
			}

		private:
			int feature_map_upsampling_size;
			int entry_upsampling_size;
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> upsampling_sizes;
			array_by_val<int_fastdiv, dimension_count> strides;
			unsigned int forward_packed_config_count;
			bool nonunit_window_x;
		};
	}
}
