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

#include "layer_tester_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "space_filling_curve.h"
#include "sequential_curve.h"
#include "int_fastdiv.h"

#include "../max_subsampling_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		template<int DIMENSION_COUNT,bool IS_MIN>
		__global__ void max_subsampling_tiling_kernel(
			float * __restrict output,
			const float * __restrict input,
			const packed_config<DIMENSION_COUNT> * __restrict spatial_config_list,
			const packed_config<DIMENSION_COUNT> * __restrict tiling_config_list,
			array_by_val<int, DIMENSION_COUNT> subsampling_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			int input_neuron_count_per_feature_map,
			int output_neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count,
			int spatial_config_count,
			int_fastdiv tiling_config_count)
		{
			int tiling_config_and_spatial_config_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
			int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int input_entry_id = blockIdx.z * blockDim.z + threadIdx.z;
			int spatial_config_id = tiling_config_and_spatial_config_pair_id / tiling_config_count;
			int tiling_config_id = tiling_config_and_spatial_config_pair_id - spatial_config_id * tiling_config_count;

			bool in_bounds = (input_entry_id < entry_count) && (feature_map_id < feature_map_count) && (tiling_config_id < tiling_config_count) && (spatial_config_id < spatial_config_count);
			if (in_bounds)
			{
				packed_config<DIMENSION_COUNT> spatial_conf = spatial_config_list[spatial_config_id];
				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = spatial_conf.get_val(i);

				packed_config<DIMENSION_COUNT> offsets_conf = tiling_config_list[tiling_config_id];
				int offset_xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					offset_xyzw[i] = offsets_conf.get_val(i);

				int current_input_elem_id = input_entry_id * feature_map_count + feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					current_input_elem_id = current_input_elem_id * input_sizes[i] + xyzw[i] * subsampling_sizes[i] + offset_xyzw[i];

				float res = IS_MIN ? 1.0e37F : -1.0e37F;

				for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? subsampling_sizes[3] : 1); ++input_w)
				{
					for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? subsampling_sizes[2] : 1); ++input_z)
					{
						for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? subsampling_sizes[1] : 1); ++input_y)
						{
							for(int input_x = 0; input_x < subsampling_sizes[0]; ++input_x)
							{
								float new_val = input[current_input_elem_id];
								res = IS_MIN ? min(res, new_val) : max(res, new_val);
								++current_input_elem_id;
							} // for input_x
							if (DIMENSION_COUNT > 1)
								current_input_elem_id += (input_sizes[0] - subsampling_sizes[0]);
						} // for input_y
						if (DIMENSION_COUNT > 2)
							current_input_elem_id += input_sizes[0] * (input_sizes[1] - subsampling_sizes[1]);
					} // for input_z
					if (DIMENSION_COUNT > 3)
						current_input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - subsampling_sizes[2]);
				} // for input_w

				int output_entry_id = tiling_config_count * input_entry_id + tiling_config_id;
				int offset = output_entry_id * feature_map_count + feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					offset = offset * output_sizes[i] + xyzw[i];
				output[offset] = res;
			}
		}

		template<int dimension_count>
		class max_subsampling_tiling_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			max_subsampling_tiling_layer_tester_cuda()
			{
			}

			virtual ~max_subsampling_tiling_layer_tester_cuda()
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
				const packed_config<spatial_dimension_count> * spatial_config_list = static_cast<const packed_config<spatial_dimension_count> *>((const void *)*persistent_working_data[0]);
				const packed_config<tiling_dimension_count> * tiling_config_list = static_cast<const packed_config<tiling_dimension_count> *>((const void *)*persistent_working_data[1]);

				if (entry_count % tiling_config_count != 0)
					throw neural_network_exception((boost::format("max_subsampling_tiling_layer_tester_cuda: entry_count (%1%) is not evenly divisible by total_tiling_factor (%2%)") % entry_count % (int)tiling_config_count).str());
				int input_entry_count = entry_count / tiling_config_count;

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					tiling_config_count * spatial_config_count,
					input_configuration_specific_list[0].feature_map_count,
					input_entry_count);

				if (is_min)
					max_subsampling_tiling_kernel<dimension_count,true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*output_buffer,
						*input_buffers[0],
						spatial_config_list,
						tiling_config_list,
						subsampling_sizes,
						input_sizes,
						output_sizes,
						input_elem_count_per_feature_map_list[0],
						output_elem_count_per_feature_map,
						output_configuration_specific.feature_map_count,
						input_entry_count,
						spatial_config_count,
						tiling_config_count);
				else
					max_subsampling_tiling_kernel<dimension_count,false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*output_buffer,
						*input_buffers[0],
						spatial_config_list,
						tiling_config_list,
						subsampling_sizes,
						input_sizes,
						output_sizes,
						input_elem_count_per_feature_map_list[0],
						output_elem_count_per_feature_map,
						output_configuration_specific.feature_map_count,
						input_entry_count,
						spatial_config_count,
						tiling_config_count);
			}

		protected:
			static const int spatial_dimension_count = dimension_count;
			static const int tiling_dimension_count = dimension_count;

			virtual void tester_configured()
			{
				nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

				is_min = layer_derived->is_min;

				for(int i = 0; i < dimension_count; ++i)
				{
					subsampling_sizes[i] = layer_derived->subsampling_sizes[i];
					input_sizes[i] = input_configuration_specific_list[0].dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];
				}

				spatial_config_count = 1;
				for(int i = 0; i < dimension_count; ++i)
					spatial_config_count *= output_sizes[i];

				unsigned int t1 = 1;
				for(int i = 0; i < dimension_count; ++i)
					t1 *= subsampling_sizes[i];
				tiling_config_count = t1;
			}

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_persistent_working_data() const
			{
				std::vector<cuda_linear_buffer_device::const_ptr> res;

				{
					std::vector<packed_config<spatial_dimension_count> > task_list;
					{
						nnforge_array<int, dimension_count> size_list;
						for(int i = 0; i < dimension_count; ++i)
							size_list[i] = output_sizes[i];
						std::vector<nnforge_array<int, dimension_count> > ordered_list;
						sequential_curve<dimension_count>::fill_pattern(size_list, ordered_list);
						packed_config<spatial_dimension_count> new_elem;
						for(int j = 0; j < ordered_list.size(); ++j)
						{
							const nnforge_array<int, dimension_count>& spatial_dimensions = ordered_list[j];
							for(int i = 0; i < dimension_count; ++i)
								new_elem.set_val(i, spatial_dimensions[i]);
							task_list.push_back(new_elem);
						}
					}
					res.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(&task_list[0], sizeof(packed_config<spatial_dimension_count>) * task_list.size())));
				}

				{
					std::vector<packed_config<tiling_dimension_count> > task_list;
					{
						nnforge_array<int, dimension_count> subsampling_list;
						for(int i = 0; i < dimension_count; ++i)
							subsampling_list[i] = subsampling_sizes[i];
						std::vector<nnforge_array<int, dimension_count> > ordered_list;
						sequential_curve<dimension_count>::fill_pattern(subsampling_list, ordered_list);
						packed_config<tiling_dimension_count> new_elem;
						for(int j = 0; j < ordered_list.size(); ++j)
						{
							const nnforge_array<int, dimension_count>& subsampling_offsets = ordered_list[j];
							for(int i = 0; i < dimension_count; ++i)
								new_elem.set_val(i, subsampling_offsets[i]);
							task_list.push_back(new_elem);
						}
					}
					res.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(&task_list[0], sizeof(packed_config<tiling_dimension_count>) * task_list.size())));
				}

				return res;
			}

		private:
			bool is_min;
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> subsampling_sizes;

			unsigned int spatial_config_count;
			int_fastdiv tiling_config_count;
		};
	}
}
