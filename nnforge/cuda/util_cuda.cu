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
#include "../layer_configuration_specific.h"

#include <boost/format.hpp>

#include <utility>

namespace nnforge
{
	namespace cuda
	{
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

		__global__ void set_with_value_util_kernel(
			int4 * __restrict buf,
			int v,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				int4 val;
				val.x = v;
				val.y = v;
				val.z = v;
				val.w = v;
				buf[elem_id] = val;
			}
		}

		__global__ void set_with_value_util_kernel(
			double2 * __restrict buf,
			double v,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				double2 val;
				val.x = v;
				val.y = v;
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

		__global__ void apply_weight_decay_util_kernel(
			const float4 * __restrict learning_rates,
			float4 * __restrict weights,
			float weight_decay,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = learning_rates[elem_id];
				float4 current_weight = weights[elem_id];
				val.x = 1.0F - val.x * weight_decay;
				val.y = 1.0F - val.y * weight_decay;
				val.z = 1.0F - val.z * weight_decay;
				val.w = 1.0F - val.w * weight_decay;
				current_weight.x *= val.x;
				current_weight.y *= val.y;
				current_weight.z *= val.z;
				current_weight.w *= val.w;
				weights[elem_id] = current_weight;
			}
		}

		__global__ void apply_gradient_with_weight_decay_util_kernel(
			const float2 * __restrict gradient,
			const float2 * __restrict learning_rates,
			float2 * __restrict weights,
			float weight_decay,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float2 lr = learning_rates[elem_id];
				float2 current_weight = weights[elem_id];
				float2 grad = gradient[elem_id];
				float2 new_weight;
				new_weight.x = current_weight.x + lr.x * (grad.x - weight_decay * current_weight.x);
				new_weight.y = current_weight.y + lr.y * (grad.y - weight_decay * current_weight.y);
				weights[elem_id] = new_weight;
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

		__global__ void copy_to_striped_kernel(
			const float * __restrict source_buf,
			float2 * __restrict dest_buf,
			int elem_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int strided_feature_map_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;

			int first_feature_map_id = strided_feature_map_id * 2;
			if ((elem_id < elem_count_per_feature_map) && (first_feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				int tt = entry_id * elem_count_per_feature_map;
				int base_src_offset = tt * feature_map_count + elem_id;
				int base_dst_offset = tt * ((feature_map_count + 1) >> 1) + elem_id;
				float2 pack;
				pack.x = source_buf[first_feature_map_id * elem_count_per_feature_map + base_src_offset];
				pack.y = 0.0F;
				int second_feature_map_id = first_feature_map_id + 1;
				if (second_feature_map_id < feature_map_count)
					pack.y = source_buf[second_feature_map_id * elem_count_per_feature_map + base_src_offset];

				dest_buf[strided_feature_map_id * elem_count_per_feature_map + base_dst_offset] = pack;
			}
		}

		__global__ void copy_from_striped_kernel(
			const float2 * __restrict source_buf,
			float * __restrict dest_buf,
			int elem_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			int strided_feature_map_id = blockDim.y * blockIdx.y + threadIdx.y;
			int entry_id = blockDim.z * blockIdx.z + threadIdx.z;

			int first_feature_map_id = strided_feature_map_id * 2;
			if ((elem_id < elem_count_per_feature_map) && (first_feature_map_id < feature_map_count) && (entry_id < entry_count))
			{
				int tt = entry_id * elem_count_per_feature_map;
				int base_dst_offset = tt * feature_map_count + elem_id;
				int base_src_offset = tt * ((feature_map_count + 1) >> 1) + elem_id;

				float2 pack = source_buf[strided_feature_map_id * elem_count_per_feature_map + base_src_offset];

				dest_buf[first_feature_map_id * elem_count_per_feature_map + base_dst_offset] = pack.x;
				int second_feature_map_id = first_feature_map_id + 1;
				if (second_feature_map_id < feature_map_count)
					dest_buf[second_feature_map_id * elem_count_per_feature_map + base_dst_offset] = pack.y;
			}
		}

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

		void cuda_util::set_with_value(
			const cuda_running_configuration& cuda_config,
			double * buf_with_aligned_size,
			double v,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 1) / 2;
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			set_with_value_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((double2 *)buf_with_aligned_size, v, new_elem_count);
		}

		void cuda_util::set_with_value(
			const cuda_running_configuration& cuda_config,
			int * buf_with_aligned_size,
			int v,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			set_with_value_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((int4 *)buf_with_aligned_size, v, new_elem_count);
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

		void cuda_util::apply_weight_decay(
			const cuda_running_configuration& cuda_config,
			const float * learning_rates_with_aligned_size,
			float * weights_with_aligned_size,
			float weight_decay,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			if (weight_decay != 0.0F)
			{
				int new_elem_count = (elem_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					cuda_config,
					new_elem_count);
				apply_weight_decay_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((const float4 *)learning_rates_with_aligned_size, (float4 *)weights_with_aligned_size, weight_decay, new_elem_count);
			}
		}

		void cuda_util::apply_gradient_with_weight_decay(
			const cuda_running_configuration& cuda_config,
			const float * gradient_with_aligned_size,
			const float * learning_rates_with_aligned_size,
			float * weights_with_aligned_size,
			float weight_decay,
			int elem_count,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 1) / 2;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			apply_gradient_with_weight_decay_util_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((const float2 *)gradient_with_aligned_size, (const float2 *)learning_rates_with_aligned_size, (float2 *)weights_with_aligned_size, weight_decay, new_elem_count);
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
			int divisible)
		{
			const int assume_threadblock_size = 256;
			int threadblock_count = (total_thread_count + assume_threadblock_size - 1) / assume_threadblock_size;
			float wave_count = static_cast<float>(threadblock_count) / static_cast<float>(cuda_config.multiprocessor_count * 4);
			if (wave_count >= 4.0F)
				return 1;

			int current_div;
			for(int wave_count = 1; wave_count <= 4; ++wave_count)
			{
				current_div = std::min((cuda_config.multiprocessor_count * 4 * wave_count * assume_threadblock_size) / total_thread_count, divisible);
				if (current_div == 0)
					continue;
				int group_size = (divisible + current_div - 1) / current_div;
				current_div = (divisible + group_size - 1) / group_size;
				int current_threadblock_count = (total_thread_count * current_div + assume_threadblock_size - 1) / assume_threadblock_size;
				float current_wave_count = static_cast<float>(current_threadblock_count) / static_cast<float>(cuda_config.multiprocessor_count * 4);
				float remaining_part = wave_count - current_wave_count;
				if (remaining_part < 0.2F)
					return current_div;
			}

			return std::min(std::max(current_div, 1), divisible);
		}

		int cuda_util::get_thread_count_per_wave(const cuda_running_configuration& cuda_config)
		{
			return cuda_config.multiprocessor_count * 4 * 256;
		}

		unsigned int cuda_util::get_feature_map_count_striped(unsigned int feature_map_count)
		{
			return ((feature_map_count + 1) >> 1);
		}

		layer_configuration_specific cuda_util::get_layer_configuration_specific_striped(const layer_configuration_specific& original_layer_config)
		{
			layer_configuration_specific res = original_layer_config;
			res.feature_map_count = get_feature_map_count_striped(res.feature_map_count);
			return res;
		}

		void cuda_util::copy_to_striped(
			const cuda_running_configuration& cuda_config,
			const float * source_buf,
			float2 * dest_buf,
			unsigned int elem_count_per_feature_map,
			unsigned int feature_map_count,
			unsigned int entry_count,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count_per_feature_map,
				get_feature_map_count_striped(feature_map_count),
				entry_count);
			copy_to_striped_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(source_buf, dest_buf, elem_count_per_feature_map, feature_map_count, entry_count);
		}

		void cuda_util::copy_from_striped(
			const cuda_running_configuration& cuda_config,
			const float2 * source_buf,
			float * dest_buf,
			unsigned int elem_count_per_feature_map,
			unsigned int feature_map_count,
			unsigned int entry_count,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count_per_feature_map,
				get_feature_map_count_striped(feature_map_count),
				entry_count);
			copy_from_striped_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(source_buf, dest_buf, elem_count_per_feature_map, feature_map_count, entry_count);
		}
	}
}
