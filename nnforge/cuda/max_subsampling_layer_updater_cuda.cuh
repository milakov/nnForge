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

#pragma once

#include "layer_updater_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "cuda_texture.h"
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "space_filling_curve.h"
#include "sequential_curve.h"

#include "../max_subsampling_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		#define FEATURE_MAP_BLOCK_SIZE 4

		extern __shared__ float arr_sh[];

		template<int DIMENSION_COUNT>
		__global__ void max_subsampling_upd_kernel(
			float * __restrict output,
			int * __restrict max_positions,
			const float * __restrict input,
			const packed_config<DIMENSION_COUNT+1> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> subsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			int local_thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
			int threadblock_size = blockDim.z * blockDim.y * blockDim.x;

			float * vals = arr_sh;
			int * max_pos_list = (int *)(vals + threadblock_size * FEATURE_MAP_BLOCK_SIZE);

			bool in_bounds = (entry_id < entry_count) && (base_feature_map_id < feature_map_count) && (packed_config_id < packed_config_count);

			float res[FEATURE_MAP_BLOCK_SIZE];
			bool item_valid[FEATURE_MAP_BLOCK_SIZE - 1];
			int max_pos[FEATURE_MAP_BLOCK_SIZE];
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
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					item_valid[i - 1] = (base_feature_map_id + i < feature_map_count);

				bool init_required = true;
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
								{
									if (init_required || (new_val[i] > res[i]))
									{
										res[i] = new_val[i];
										max_pos[i] = current_input_elem_id + input_neuron_count_per_feature_map * i;
									}
								}
								current_input_elem_id += input_sizes[0];
								init_required = false;
							}
							else
							{
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								{
									res[i] = new_val[i];
									max_pos[i] = current_input_elem_id + input_neuron_count_per_feature_map * i;
								}
							}
						} // for input_y
						current_input_elem_id += input_sizes[0] * (input_sizes[1] - subsampling_sizes[1]);
					} // for input_z
					current_input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - subsampling_sizes[2]);
				} // for input_w

				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					vals[local_thread_id + threadblock_size * i] = res[i];
					max_pos_list[local_thread_id + threadblock_size * i] = max_pos[i];
				}
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
						int new_pos = max_pos_list[local_thread_id + threadblock_size * i];

						if (new_val > res[i])
						{
							res[i] = new_val;
							max_pos[i] = new_pos;
						}
					}
				}
				int offset = entry_id * feature_map_count + base_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					offset = offset * output_sizes[i] + xyzw[i];
				output[offset] = res[0];
				max_positions[offset] = max_pos[0];
				#pragma unroll
				for(int i = 1; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					offset += output_neuron_count_per_feature_map;
					if (item_valid[i - 1])
					{
						output[offset] = res[i];
						max_positions[offset] = max_pos[i];
					}
				}
			}
		}

		__global__ void max_subsampling_backprop_upd_kernel(
			float * __restrict input_erros,
			const int * __restrict max_positions,
			const float * __restrict output_errors,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
				input_erros[max_positions[elem_id]] = output_errors[elem_id];
		}

		template<int dimension_count>
		class max_subsampling_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			max_subsampling_layer_updater_cuda()
			{
			}

			virtual ~max_subsampling_layer_updater_cuda()
			{
			}

			virtual void enqueue_test(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count)
			{
				if (offset_input_entry_id > 0)
					throw neural_network_exception("max_subsampling_layer_updater_cuda is not able to run using offset");

				const float * input = *input_neurons_buffer;
				float * output = *output_neurons_buffer;
				int * max_positions = (int *)((void *)(*additional_buffers[0]));
				const packed_config<forward_dimension_count> * packed_config_list = static_cast<const packed_config<forward_dimension_count> *>((const void *)*additional_buffers[1]);

				int feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					feature_map_block_count,
					entry_count,
					subsampling_sizes[0]);

				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * (sizeof(float) + sizeof(int)) * FEATURE_MAP_BLOCK_SIZE;

				max_subsampling_upd_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					output,
					max_positions,
					input,
					packed_config_list,
					subsampling_sizes,
					input_sizes,
					output_sizes,
					input_elem_count_per_feature_map,
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					forward_packed_config_count);
			}

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
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

				const float * output_errors = *output_errors_buffer;
				float * input_errors = *input_errors_buffer;
				const int * max_positions = (const int *)((const void *)(*additional_buffers[0]));

				int elem_count = output_elem_count_per_entry * entry_count;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);

				max_subsampling_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					input_errors,
					max_positions,
					output_errors,
					elem_count);
			}

		protected:
			static const int forward_dimension_count = (dimension_count + 1);

			virtual void updater_configured()
			{
				nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

				for(int i = 0; i < dimension_count; ++i)
				{
					subsampling_sizes[i] = layer_derived->subsampling_sizes[i];
					input_sizes[i] = input_configuration_specific.dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];
				}

				forward_packed_config_count = subsampling_sizes[0];
				for(int i = 0; i < dimension_count; ++i)
					forward_packed_config_count *= output_sizes[i];
			}

			virtual bool is_in_place_backprop() const
			{
				return false;
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const
			{
				std::vector<size_t> res;

				res.push_back(output_elem_count_per_entry * sizeof(float));

				return res;
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const
			{
				std::vector<size_t> res;

				res.push_back(sizeof(packed_config<forward_dimension_count>) * forward_packed_config_count);

				return res;
			}

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
			{
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
				cuda_safe_call(cudaMemcpy(*additional_buffers[1], &(*task_list.begin()), sizeof(packed_config<forward_dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
			}

		private:
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> subsampling_sizes;

			unsigned int forward_packed_config_count;
		};
	}
}
