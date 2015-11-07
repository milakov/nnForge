/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include <boost/format.hpp>

#include "util_cuda.h"
#include "cuda_texture.h"
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "sequential_curve.h"

#include "../average_subsampling_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		#define FEATURE_MAP_BLOCK_SIZE 4

		extern __shared__ float arr_sh[];

		template<int DIMENSION_COUNT>
		__global__ void average_subsampling_upd_kernel(
			float * __restrict output,
			const float * __restrict input,
			const packed_config<DIMENSION_COUNT+1> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> subsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count,
			int packed_config_count,
			float mult)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			int local_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
			int threadblock_size = blockDim.z * blockDim.y * blockDim.x;

			float * vals = arr_sh;

			bool in_bounds = (entry_id < entry_count) && (base_feature_map_id < feature_map_count) && (packed_config_id < packed_config_count);

			float res[FEATURE_MAP_BLOCK_SIZE];
			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			int window_x;
			int xyzw[DIMENSION_COUNT];
			if (in_bounds)
			{
				packed_config<DIMENSION_COUNT+1> conf = packed_config_list[packed_config_id];
				
				window_x = conf.get_val(0);
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i + 1);

				int current_input_elem_id = entry_id * feature_map_count + base_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					current_input_elem_id = current_input_elem_id * input_sizes[i] + xyzw[i] * subsampling_sizes[i];
				current_input_elem_id += window_x;

				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					res[i] = 0.0F;
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					item_valid[i - 1] = (base_feature_map_id + i < feature_map_count);

				for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? subsampling_sizes[3] : 1); ++input_w)
				{
					for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? subsampling_sizes[2] : 1); ++input_z)
					{
						for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? subsampling_sizes[1] : 1); ++input_y)
						{
							float new_val[FEATURE_MAP_BLOCK_SIZE];
							new_val[0] = input[current_input_elem_id];
							#pragma unroll
							for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								if (item_valid[i - 1])
									new_val[i] = input[current_input_elem_id + input_neuron_count_per_feature_map * i];
							if (DIMENSION_COUNT > 1)
							{
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									res[i] += new_val[i];
								current_input_elem_id += input_sizes[0];
							}
							else
							{
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									res[i] = new_val[i];
							}
						} // for input_y
						current_input_elem_id += input_sizes[0] * (input_sizes[1] - subsampling_sizes[1]);
					} // for input_z
					current_input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - subsampling_sizes[2]);
				} // for input_w

				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					vals[local_thread_id + threadblock_size * i] = res[i];
			}

			__syncthreads();

			if (in_bounds && (window_x == 0))
			{
				for(int j = 1; j < subsampling_sizes[0]; ++j)
				{
					local_thread_id++;
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						float new_val = vals[local_thread_id + threadblock_size * i];
						res[i] += new_val;
					}
				}
				int offset = entry_id * feature_map_count + base_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					offset = offset * output_sizes[i] + xyzw[i];
				output[offset] = res[0] * mult;
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					offset += output_neuron_count_per_feature_map;
					if (item_valid[i - 1])
						output[offset] = res[i] * mult;
				}
			}
		}

		template<int DIMENSION_COUNT>
		__global__ void average_subsampling_backprop_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
			const packed_config<DIMENSION_COUNT+1> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> subsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count,
			int packed_config_count,
			float mult,
			bool add_update_to_destination)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (base_feature_map_id < feature_map_count) && (packed_config_id < packed_config_count);
			if (!in_bounds)
				return;

			int window_x;
			int xyzw[DIMENSION_COUNT];
			packed_config<DIMENSION_COUNT+1> conf = packed_config_list[packed_config_id];
			window_x = conf.get_val(0);
			#pragma unroll
			for(int i = 0; i < DIMENSION_COUNT; ++i)
				xyzw[i] = conf.get_val(i + 1);

			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			#pragma unroll
			for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				item_valid[i - 1] = (base_feature_map_id + i < feature_map_count);

			float err[FEATURE_MAP_BLOCK_SIZE];
			int offset = entry_id * feature_map_count + base_feature_map_id;
			#pragma unroll
			for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				offset = offset * output_sizes[i] + xyzw[i];

			err[0] = output_errors[offset] * mult;
			#pragma unroll
			for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
			{
				offset += output_neuron_count_per_feature_map;
				if (item_valid[i - 1])
					err[i] = output_errors[offset] * mult;
			}

			int current_input_elem_id = entry_id * feature_map_count + base_feature_map_id;
			#pragma unroll
			for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
				current_input_elem_id = current_input_elem_id * input_sizes[i] + xyzw[i] * subsampling_sizes[i];
			current_input_elem_id += window_x;

			for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? subsampling_sizes[3] : 1); ++input_w)
			{
				for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? subsampling_sizes[2] : 1); ++input_z)
				{
					for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? subsampling_sizes[1] : 1); ++input_y)
					{
						if (add_update_to_destination)
						{
							float dst[FEATURE_MAP_BLOCK_SIZE];
							dst[0] = input_errors[current_input_elem_id];
							#pragma unroll
							for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								if (item_valid[i - 1])
									dst[i] = input_errors[current_input_elem_id + input_neuron_count_per_feature_map * i];
							input_errors[current_input_elem_id] = err[0];
							#pragma unroll
							for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								if (item_valid[i - 1])
									input_errors[current_input_elem_id + input_neuron_count_per_feature_map * i] = err[i] + dst[i];
							if (DIMENSION_COUNT > 1)
							{
								current_input_elem_id += input_sizes[0];
							}
						}
						else
						{
							input_errors[current_input_elem_id] = err[0];
							#pragma unroll
							for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								if (item_valid[i - 1])
									input_errors[current_input_elem_id + input_neuron_count_per_feature_map * i] = err[i];
							if (DIMENSION_COUNT > 1)
							{
								current_input_elem_id += input_sizes[0];
							}
						}
					} // for input_y
					current_input_elem_id += input_sizes[0] * (input_sizes[1] - subsampling_sizes[1]);
				} // for input_z
				current_input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - subsampling_sizes[2]);
			} // for input_w
		}

		template<int dimension_count>
		class average_subsampling_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			average_subsampling_layer_updater_cuda()
			{
			}

			virtual ~average_subsampling_layer_updater_cuda()
			{
			}

			virtual void enqueue_forward_propagation(
				cudaStream_t stream_id,
				cuda_linear_buffer_device::ptr output_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::ptr>& data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
				unsigned int entry_count)
			{
				const packed_config<forward_dimension_count> * packed_config_list = static_cast<const packed_config<forward_dimension_count> *>((const void *)*persistent_working_data[0]);

				int feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					subsampling_sizes[0]);

				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float) * FEATURE_MAP_BLOCK_SIZE;

				average_subsampling_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					packed_config_list,
					subsampling_sizes,
					input_sizes,
					output_sizes,
					input_elem_count_per_feature_map_list[0],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					forward_packed_config_count,
					mult);
			}

			virtual void enqueue_backward_data_propagation(
				cudaStream_t stream_id,
				unsigned int input_index,
				cuda_linear_buffer_device::ptr input_errors_buffer,
				cuda_linear_buffer_device::const_ptr output_errors_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::ptr>& data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
				cuda_linear_buffer_device::const_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
				bool add_update_to_destination,
				unsigned int entry_count)
			{
				if ((!is_even_subsampling) && (!add_update_to_destination))
				{
					cuda_util::set_with_value(
						*cuda_config,
						*input_errors_buffer,
						0.0F,
						input_elem_count_per_entry_list[0] * entry_count,
						stream_id);
				}

				const packed_config<forward_dimension_count> * packed_config_list = static_cast<const packed_config<forward_dimension_count> *>((const void *)*persistent_working_data[0]);

				int feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					subsampling_sizes[0]);

				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float) * FEATURE_MAP_BLOCK_SIZE;

				average_subsampling_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*input_errors_buffer,
					*output_errors_buffer,
					packed_config_list,
					subsampling_sizes,
					input_sizes,
					output_sizes,
					input_elem_count_per_feature_map_list[0],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					forward_packed_config_count,
					mult,
					add_update_to_destination);
			}

		protected:
			static const int forward_dimension_count = (dimension_count + 1);

			virtual void updater_configured()
			{
				nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);

				unsigned int toal_subsampling_size = 1;
				is_even_subsampling = true;
				for(int i = 0; i < dimension_count; ++i)
				{
					subsampling_sizes[i] = layer_derived->subsampling_sizes[i];
					input_sizes[i] = input_configuration_specific_list[0].dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];

					toal_subsampling_size *= subsampling_sizes[i];
					is_even_subsampling = is_even_subsampling & (input_sizes[i] == output_sizes[i] * subsampling_sizes[i]);
				}

				mult = 1.0F / static_cast<float>(toal_subsampling_size);

				forward_packed_config_count = subsampling_sizes[0];
				for(int i = 0; i < dimension_count; ++i)
					forward_packed_config_count *= output_sizes[i];
			}

			bool is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
			{
				return false;
			}

			bool is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
			{
				return false;
			}

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_persistent_working_data() const
			{
				std::vector<cuda_linear_buffer_device::const_ptr> res;

				std::vector<packed_config<forward_dimension_count> > task_list;
				{
					nnforge_array<int, dimension_count> size_list;
					for(int i = 0; i < dimension_count; ++i)
						size_list[i] = output_sizes[i];
					std::vector<nnforge_array<int, dimension_count> > ordered_list;
					sequential_curve<dimension_count>::fill_pattern(size_list, ordered_list);
					packed_config<forward_dimension_count> new_elem;
					for(int j = 0; j < ordered_list.size(); ++j)
					{
						const nnforge_array<int, dimension_count>& spatial_dimensions = ordered_list[j];
						for(int i = 0; i < dimension_count; ++i)
							new_elem.set_val(i + 1, spatial_dimensions[i]);
						for(int k = 0; k < subsampling_sizes[0]; ++k)
						{
							new_elem.set_val(0, k);
							task_list.push_back(new_elem);
						}
					}
				}

				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(&task_list[0], sizeof(packed_config<forward_dimension_count>) * task_list.size())));

				return res;
			}

		private:
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> subsampling_sizes;

			unsigned int forward_packed_config_count;
			float mult;
			bool is_even_subsampling;
		};
	}
}
