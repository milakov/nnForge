/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include <stack>

#include "cuda_running_configuration.h"

static __forceinline__ __device__ float __load_nc(const float * ptr)
{
#if __CUDA_ARCH__ >= 350
	return __ldg(ptr);
#else
	return *ptr;
#endif
}

namespace nnforge
{
	namespace cuda
	{
		class cuda_util
		{
		public:
			static std::pair<dim3, dim3> get_grid_and_threadblock_sizes_2d_access(
				const cuda_running_configuration& cuda_config,
				unsigned int x,
				unsigned int y,
				unsigned int z);

			static std::pair<dim3, dim3> get_grid_and_threadblock_sizes_2d_access_x_aligned(
				const cuda_running_configuration& cuda_config,
				unsigned int x,
				unsigned int y,
				unsigned int z);

			static std::pair<dim3, dim3> get_grid_and_threadblock_sizes_sequential_access(
				const cuda_running_configuration& cuda_config,
				unsigned int x,
				unsigned int y,
				unsigned int z,
				unsigned int threadblock_size_x_evenly_divisible = 1);

			static std::pair<dim3, dim3> get_grid_and_threadblock_sizes_sequential_access(
				const cuda_running_configuration& cuda_config,
				int elem_count);

			static int get_power2_aligned_size(int original_size);

			static int reinterpret(float val)
			{
				union {int int_val; float float_val;} v;
				v.float_val = val;
				return v.int_val;
			}

			static size_t get_float4_aligned_buffer_size(size_t original_size);

			static void set_with_value(
				const cuda_running_configuration& cuda_config,
				float * buf_with_aligned_size,
				float v,
				int elem_count,
				cudaStream_t cuda_stream);

			static void multiply_by_value(
				const cuda_running_configuration& cuda_config,
				float * buf_with_aligned_size,
				float v,
				int elem_count,
				cudaStream_t cuda_stream);

			static void multiply_by_itself(
				const cuda_running_configuration& cuda_config,
				const float * input_buf_with_aligned_size,
				float * output_buf_with_aligned_size,
				int elem_count,
				cudaStream_t cuda_stream);

			static void copy_buffer(
				const cuda_running_configuration& cuda_config,
				const float * input_buf_with_aligned_size,
				float * output_with_aligned_size,
				int elem_count,
				cudaStream_t cuda_stream);

			static int get_group_count(
				const cuda_running_configuration& cuda_config,
				int total_thread_count,
				int divisible,
				bool more_threadblocks = false);

			static void fill_tiling_pattern(
				int size_x,
				int size_y,
				std::vector<std::pair<int, int> >& pair_list);

		private:
			cuda_util();
			cuda_util(const cuda_util&);
			cuda_util& operator =(const cuda_util&);
			~cuda_util();

			struct tile
			{
				tile(int left_x, int right_x, int top_y, int bottom_y);

				bool is_point() const;

				void split_to_stack(
					std::stack<tile>& st,
					int start_x,
					int start_y) const;

				int left_x;
				int right_x;
				int top_y;
				int bottom_y;
			};

			static const unsigned int preferred_width_2d_access;
			static const unsigned int preferred_height_2d_access;
			static const unsigned int preferred_threadblocksize_sequential_access;
			static const unsigned int preferred_width_2d_access_x_aligned;
			static const unsigned int preferred_height_2d_access_x_aligned;
		};
	}
}
