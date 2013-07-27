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

#include "util_cuda.h"
#include "../neural_network_exception.h"

#include <boost/format.hpp>

#include <utility>

__global__ void set_with_value_util_kernel(
	float4 * __restrict buf,
	float v,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val;
		val.x = v;
		val.y = v;
		val.z = v;
		val.w = v;
		buf[elem_id] = val;
	}
}

__global__ void multiply_by_value_util_kernel(
	float4 * __restrict buf,
	float v,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = buf[elem_id];
		val.x *= v;
		val.y *= v;
		val.z *= v;
		val.w *= v;
		buf[elem_id] = val;
	}
}

__global__ void multiply_by_itself_training_util_kernel(
	const float4 * __restrict input_buf,
	float4 * __restrict output_buf,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = input_buf[elem_id];
		val.x *= val.x;
		val.y *= val.y;
		val.z *= val.z;
		val.w *= val.w;
		output_buf[elem_id] = val;
	}
}

__global__ void copy_buffer_util_kernel(
	const float4 * __restrict input_buf,
	float4 * __restrict output_buf,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
		output_buf[elem_id] = input_buf[elem_id];
}

namespace nnforge
{
	namespace cuda
	{
		const unsigned int cuda_util::preferred_width_2d_access = 16;
		const unsigned int cuda_util::preferred_height_2d_access = 16;
		const unsigned int cuda_util::preferred_threadblocksize_sequential_access = 256;
		const unsigned int cuda_util::preferred_width_2d_access_x_aligned = 32;
		const unsigned int cuda_util::preferred_height_2d_access_x_aligned = 8;

		std::pair<dim3, dim3> cuda_util::get_grid_and_threadblock_sizes_2d_access(
			const cuda_running_configuration& cuda_config,
			unsigned int x,
			unsigned int y,
			unsigned int z)
		{
			dim3 threadblock_size(1, 1, 1);

			const unsigned int preferred_threadblock_size = preferred_width_2d_access * preferred_height_2d_access;

			if (x < preferred_width_2d_access)
			{
				threadblock_size.x = x;
				threadblock_size.y = std::min<unsigned int>(cuda_config.max_threads_dim[1], std::min<unsigned int>(y, preferred_threadblock_size / threadblock_size.x));
			}
			else
			{
				if (y < preferred_height_2d_access)
				{
					threadblock_size.y = y;
					threadblock_size.x = std::min<unsigned int>(cuda_config.max_threads_dim[0], std::min<unsigned int>(x, preferred_threadblock_size / threadblock_size.y));
				}
				else
				{
					threadblock_size.x = preferred_width_2d_access;
					threadblock_size.y = preferred_height_2d_access;
				}
			}


			unsigned int threadblocks_to_cover_x = (x + threadblock_size.x - 1) / threadblock_size.x;
			threadblock_size.x = (x + threadblocks_to_cover_x - 1) / threadblocks_to_cover_x;
			unsigned int threadblocks_to_cover_y = (y + threadblock_size.y - 1) / threadblock_size.y;
			threadblock_size.y = (y + threadblocks_to_cover_y - 1) / threadblocks_to_cover_y;

			threadblock_size.z = std::min<unsigned int>(cuda_config.max_threads_dim[2], std::min<unsigned int>(z, preferred_threadblock_size / (threadblock_size.x * threadblock_size.y)));
			unsigned int threadblocks_to_cover_z = (z + threadblock_size.z - 1) / threadblock_size.z;
			threadblock_size.z = (z + threadblocks_to_cover_z - 1) / threadblocks_to_cover_z;

			dim3 grid_size(
				(x + threadblock_size.x - 1) / threadblock_size.x,
				(y + threadblock_size.y - 1) / threadblock_size.y,
				(z + threadblock_size.z - 1) / threadblock_size.z);

			return std::make_pair(grid_size, threadblock_size);
		}

		std::pair<dim3, dim3> cuda_util::get_grid_and_threadblock_sizes_2d_access_x_aligned(
			const cuda_running_configuration& cuda_config,
			unsigned int x,
			unsigned int y,
			unsigned int z)
		{
			dim3 threadblock_size(1, 1, 1);

			const unsigned int preferred_threadblock_size = preferred_width_2d_access_x_aligned * preferred_height_2d_access_x_aligned;

			if (x < preferred_width_2d_access_x_aligned)
			{
				threadblock_size.x = x;
				threadblock_size.y = std::min<unsigned int>(cuda_config.max_threads_dim[1], std::min<unsigned int>(y, preferred_threadblock_size / threadblock_size.x));
			}
			else
			{
				if (y < preferred_height_2d_access_x_aligned)
				{
					threadblock_size.y = y;
					threadblock_size.x = std::min<unsigned int>(cuda_config.max_threads_dim[0], std::min<unsigned int>(x, preferred_threadblock_size / threadblock_size.y));
				}
				else
				{
					threadblock_size.x = preferred_width_2d_access_x_aligned;
					threadblock_size.y = preferred_height_2d_access_x_aligned;
				}
			}


			unsigned int threadblocks_to_cover_x = (x + threadblock_size.x - 1) / threadblock_size.x;
			threadblock_size.x = (x + threadblocks_to_cover_x - 1) / threadblocks_to_cover_x;
			unsigned int threadblocks_to_cover_y = (y + threadblock_size.y - 1) / threadblock_size.y;
			threadblock_size.y = (y + threadblocks_to_cover_y - 1) / threadblocks_to_cover_y;

			threadblock_size.z = std::min<unsigned int>(cuda_config.max_threads_dim[2], std::min<unsigned int>(z, preferred_threadblock_size / (threadblock_size.x * threadblock_size.y)));
			unsigned int threadblocks_to_cover_z = (z + threadblock_size.z - 1) / threadblock_size.z;
			threadblock_size.z = (z + threadblocks_to_cover_z - 1) / threadblocks_to_cover_z;

			dim3 grid_size(
				(x + threadblock_size.x - 1) / threadblock_size.x,
				(y + threadblock_size.y - 1) / threadblock_size.y,
				(z + threadblock_size.z - 1) / threadblock_size.z);

			return std::make_pair(grid_size, threadblock_size);
		}

		std::pair<dim3, dim3> cuda_util::get_grid_and_threadblock_sizes_sequential_access(
			const cuda_running_configuration& cuda_config,
			unsigned int x,
			unsigned int y,
			unsigned int z,
			unsigned int threadblock_size_x_evenly_divisible)
		{
			dim3 threadblock_size(1, 1, 1);

			int max_threads_dim_x = cuda_config.max_threads_dim[0];

			unsigned int preferred_threadblock_size_remained = preferred_threadblocksize_sequential_access;

			preferred_threadblock_size_remained /= threadblock_size_x_evenly_divisible;
			if (preferred_threadblock_size_remained == 0)
			{
				if (threadblock_size_x_evenly_divisible <= cuda_config.max_threads_dim[0])
					preferred_threadblock_size_remained = 1;
				else
					throw neural_network_exception((boost::format("Too large threadblock_size_x_evenly_divisible %1%, unable to compose threabblock") % threadblock_size_x_evenly_divisible).str());
			}
			x = (x + threadblock_size_x_evenly_divisible - 1) / threadblock_size_x_evenly_divisible;
			max_threads_dim_x = max_threads_dim_x / threadblock_size_x_evenly_divisible;

			threadblock_size.x = std::min<unsigned int>(std::min<unsigned int>(x, preferred_threadblock_size_remained), max_threads_dim_x);
			unsigned int threadblocks_to_cover_x = (x + threadblock_size.x - 1) / threadblock_size.x;
			threadblock_size.x = (x + threadblocks_to_cover_x - 1) / threadblocks_to_cover_x;

			preferred_threadblock_size_remained = preferred_threadblock_size_remained / threadblock_size.x;

			threadblock_size.y = std::min<unsigned int>(std::min<unsigned int>(y, preferred_threadblock_size_remained), cuda_config.max_threads_dim[1]);
			unsigned int threadblocks_to_cover_y = (y + threadblock_size.y - 1) / threadblock_size.y;
			threadblock_size.y = (y + threadblocks_to_cover_y - 1) / threadblocks_to_cover_y;

			preferred_threadblock_size_remained = preferred_threadblock_size_remained / threadblock_size.y;

			threadblock_size.z = std::min<unsigned int>(std::min<unsigned int>(z, preferred_threadblock_size_remained), cuda_config.max_threads_dim[2]);
			unsigned int threadblocks_to_cover_z = (z + threadblock_size.z - 1) / threadblock_size.z;
			threadblock_size.z = (z + threadblocks_to_cover_z - 1) / threadblocks_to_cover_z;

			dim3 grid_size(
				(x + threadblock_size.x - 1) / threadblock_size.x,
				(y + threadblock_size.y - 1) / threadblock_size.y,
				(z + threadblock_size.z - 1) / threadblock_size.z);

			threadblock_size.x *= threadblock_size_x_evenly_divisible;

			return std::make_pair(grid_size, threadblock_size);
		}

		std::pair<dim3, dim3> cuda_util::get_grid_and_threadblock_sizes_sequential_access(
			const cuda_running_configuration& cuda_config,
			int elem_count)
		{
			dim3 threadblock_size(1, 1, 1);
			dim3 grid_size(1, 1, 1);

			threadblock_size.x = std::min<unsigned int>(preferred_threadblocksize_sequential_access, elem_count);
			unsigned int threadblocks = (elem_count + threadblock_size.x - 1) / threadblock_size.x;
			if (threadblocks <= cuda_config.max_grid_size[0])
				grid_size.x = threadblocks;
			else
			{
				grid_size.y = (threadblocks + cuda_config.max_grid_size[0] - 1) / cuda_config.max_grid_size[0];
				grid_size.x = (threadblocks + grid_size.y - 1) / grid_size.y;
			}

			return std::make_pair(grid_size, threadblock_size);
		}

		int cuda_util::get_power2_aligned_size(int original_size)
		{
			int res = 1;

			while (res < original_size)
				res <<= 1;

			return res;
		}

		size_t cuda_util::get_float4_aligned_buffer_size(size_t original_size)
		{
			size_t sz = (original_size + 15) & ~15;
			return sz;
		}

		void cuda_util::set_with_value(
			const cuda_running_configuration& cuda_config,
			float * buf_with_aligned_size,
			float v,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			set_with_value_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((float4 *)buf_with_aligned_size, v, new_elem_count);
		}

		void cuda_util::multiply_by_value(
			const cuda_running_configuration& cuda_config,
			float * buf_with_aligned_size,
			float v,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			multiply_by_value_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((float4 *)buf_with_aligned_size, v, new_elem_count);
		}

		void cuda_util::multiply_by_itself(
			const cuda_running_configuration& cuda_config,
			const float * input_buf_with_aligned_size,
			float * output_buf_with_aligned_size,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			multiply_by_itself_training_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((const float4 *)input_buf_with_aligned_size, (float4 *)output_buf_with_aligned_size, new_elem_count);
		}

		void cuda_util::copy_buffer(
			const cuda_running_configuration& cuda_config,
			const float * input_buf_with_aligned_size,
			float * output_buf_with_aligned_size,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			copy_buffer_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((const float4 *)input_buf_with_aligned_size, (float4 *)output_buf_with_aligned_size, new_elem_count);
		}

		int cuda_util::get_group_count(
			const cuda_running_configuration& cuda_config,
			int total_thread_count,
			int divisible,
			bool more_threadblocks)
		{
			int initial_threadblock_count = std::max<int>(total_thread_count / 256, 1);
			int minimum_threadblock_count = cuda_config.multiprocessor_count * 4 * (more_threadblocks ? 4 : 1);

			if (initial_threadblock_count >= minimum_threadblock_count)
				return 1;

			int group_count = std::min<int>(minimum_threadblock_count / initial_threadblock_count, static_cast<int>(powf(static_cast<float>(divisible), 2.0F/3.0F)));
			int iteration_count = (divisible + group_count - 1) / group_count;
			group_count = (divisible + iteration_count - 1) / iteration_count;

			return group_count;
		}

		void cuda_util::fill_tiling_pattern(
			int size_x,
			int size_y,
			std::vector<std::pair<int, int> >& pair_list)
		{
			pair_list.clear();

			std::stack<tile> work_set;

			int size_max = std::max<int>(size_x, size_y);
			int size_aligned = 1;
			while (size_aligned < size_max)
				size_aligned <<= 1;

			work_set.push(tile(0, size_aligned, 0, size_aligned));
			int start_x = size_aligned - size_x;
			int start_y = size_aligned - size_y;

			while (!work_set.empty())
			{
				tile cur_tile = work_set.top();
				work_set.pop();

				if (cur_tile.is_point())
				{
					int x = cur_tile.left_x - start_x;
					int y = cur_tile.top_y - start_y;

					if ((x >= 0) && (y >= 0))
						pair_list.push_back(std::make_pair(x, y));
				}
				else
					cur_tile.split_to_stack(work_set, start_x, start_y);
			}

			if (pair_list.size() != size_x * size_y)
				throw neural_network_exception("Internal error when generating tiling pattern");
		}

		cuda_util::tile::tile(int left_x, int right_x, int top_y, int bottom_y)
			: left_x(left_x)
			, right_x(right_x)
			, top_y(top_y)
			, bottom_y(bottom_y)
		{
		}

		bool cuda_util::tile::is_point() const
		{
			return ((right_x - left_x) == 1) && ((bottom_y - top_y) == 1);
		}

		void cuda_util::tile::split_to_stack(
			std::stack<tile>& st,
			int start_x,
			int start_y) const
		{
			int middle_x = (left_x + right_x) >> 1;
			int middle_y = (top_y + bottom_y) >> 1;

			st.push(tile(middle_x, right_x, middle_y, bottom_y));

			if (middle_x > start_x)
				st.push(tile(left_x, middle_x, middle_y, bottom_y));

			if (middle_y > start_y)
				st.push(tile(middle_x, right_x, top_y, middle_y));

			if ((middle_x > start_x) && (middle_y > start_y))
				st.push(tile(left_x, middle_x, top_y, middle_y));
		}
	}
}
