/*
 *  Copyright 2011-2017 Maxim Milakov
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
#include "../debug_util.h"

#include "neural_network_cuda_exception.h"

#include <boost/format.hpp>

#include <utility>
#include <algorithm>

namespace nnforge
{
	namespace cuda
	{
		__forceinline__ __device__ double atomicAddD(double* address, double val)
		{
#if (__CUDA_ARCH__ >= 600)
				return atomicAdd(address, val);
#else
				unsigned long long int* address_as_ull = (unsigned long long int*)address;
				unsigned long long int old = *address_as_ull, assumed;
				do {
					assumed = old;
					old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
				} while (assumed != old);
				return __longlong_as_double(old);
#endif
		}

		__global__ void set_with_value_util_kernel(
			float4 * __restrict buf,
			float v,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				double2 val;
				val.x = v;
				val.y = v;
				buf[elem_id] = val;
			}
		}

		template<bool add_to_destination>
		__global__ void multiply_by_value_util_kernel(
			const float4 * __restrict input_buf,
			float4 * __restrict output_buf,
			float v,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input_buf[elem_id];
				if (add_to_destination)
				{
					float4 old_val = output_buf[elem_id];
					val.x = old_val.x + val.x * v;
					val.y = old_val.y + val.y * v;
					val.z = old_val.z + val.z * v;
					val.w = old_val.w + val.w * v;
				}
				else
				{
					val.x *= v;
					val.y *= v;
					val.z *= v;
					val.w *= v;
				}
				output_buf[elem_id] = val;
			}
		}

		__global__ void apply_weight_decay_util_kernel(
			const float4 * __restrict learning_rates,
			float4 * __restrict weights,
			float weight_decay,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
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

		#define TRANSPOSE_TILE_DIM 32
		#define TRANSPOSE_BLOCK_ROWS 8
		__global__ void transpose_kernel(
			const float * __restrict src,
			float * __restrict dst,
			int src_fast_dim,
			int src_slow_dim,
			int elem_count_per_entry,
			int entry_count)
		{
			int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
			int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count);

			__shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

			for(int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
			{
				if (in_bounds && ((y + j) < src_slow_dim) && (x < src_fast_dim))
					tile[threadIdx.y + j][threadIdx.x] = src[(int)(entry_id * elem_count_per_entry + (y + j) * src_fast_dim + x)];
			}

			__syncthreads();

			x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
			y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

			for(int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
			{
				if (in_bounds && ((y + j) < src_fast_dim) && (x < src_slow_dim))
					dst[(int)(entry_id * elem_count_per_entry + (y + j) * src_slow_dim + x)] = tile[threadIdx.x][threadIdx.y + j];
			}
		}

		template<bool add_to_destination>
		__global__ void transpose23_kernel(
			const float * __restrict src,
			float * __restrict dst,
			int src_dim1,
			int src_dim2,
			int src_dim3)
		{
			int elem_id1 = blockDim.x * blockIdx.x + threadIdx.x;
			int elem_id2 = blockDim.y * blockIdx.y + threadIdx.y;
			int elem_id3 = blockDim.z * blockIdx.z + threadIdx.z;
			if ((elem_id1 < src_dim1) && (elem_id2 < src_dim2) && (elem_id3 < src_dim3))
			{
				if (add_to_destination)
					dst[(elem_id2 * src_dim3 + elem_id3) * src_dim1 + elem_id1] += src[(elem_id3 * src_dim2 + elem_id2) * src_dim1 + elem_id1];
				else
					dst[(elem_id2 * src_dim3 + elem_id3) * src_dim1 + elem_id1] = src[(elem_id3 * src_dim2 + elem_id2) * src_dim1 + elem_id1];
			}
		}

		__global__ void duplicate_vector_kernel(
			const float * __restrict src,
			float * __restrict dst,
			int vector_elem_count,
			int dup_count)
		{
			int vector_elem_id = blockIdx.x * blockDim.x + threadIdx.x;
			int dup_id = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

			if (vector_elem_id < vector_elem_count)
			{
				float val = src[vector_elem_id];
				float * current_output = dst + (int)(dup_id * vector_elem_count + vector_elem_id);
				#pragma unroll
				for(int i = 0; i < 4; ++i)
				{
					if (dup_id < dup_count)
						*current_output = val;
					current_output += vector_elem_count;
					dup_id++;
				}
			}
		}

		extern __shared__ float arr_sh[];
		template<bool update_accum_flag>
		__global__ void apply_gradient_kernel(
			float * __restrict data,
			float * __restrict gradient,
			double * __restrict update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int block_id = blockIdx.y * gridDim.x + blockIdx.x;
			int elem_id = blockDim.x * block_id + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float upd = learning_rate * (gr * normalizer - current_weight * weight_decay);
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				upd_acc = fabs(upd);
			}

			if (update_accum_flag)
			{
				int thread_id = threadIdx.x;
				int lane_id = thread_id & 31;
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
					#if __CUDACC_VER_MAJOR__ < 9
					upd_acc += __shfl_down(upd_acc, tx);
					#else
					upd_acc += __shfl_down_sync(0xFFFFFFFF, upd_acc, tx);
					#endif
				}

				if (blockDim.x > 32)
				{
					if (lane_id == 0)
						arr_sh[thread_id >> 5] = upd_acc;
					__syncthreads();
				}

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						upd_acc += arr_sh[i];
					double upd_acc_d = (double)upd_acc;

					int accum_bucket_id = block_id & update_accum_mask;

					atomicAddD(update_accum + accum_bucket_id, upd_acc_d);
				}
			}
		}

		template<bool update_accum_flag>
		__global__ void apply_gradient_with_vanilla_momentum_kernel(
			float * __restrict data,
			float * __restrict gradient,
			float * __restrict previous_upd,
			double * __restrict update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int block_id = blockIdx.y * gridDim.x + blockIdx.x;
			int elem_id = blockDim.x * block_id + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float prev_upd = __load_nc(previous_upd + elem_id);
				float upd = prev_upd * momentum + learning_rate * (gr * normalizer - current_weight * weight_decay);
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				previous_upd[elem_id] = upd;
				upd_acc = fabs(upd);
			}

			if (update_accum_flag)
			{
				int thread_id = threadIdx.x;
				int lane_id = thread_id & 31;
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
					#if __CUDACC_VER_MAJOR__ < 9
					upd_acc += __shfl_down(upd_acc, tx);
					#else
					upd_acc += __shfl_down_sync(0xFFFFFFFF, upd_acc, tx);
					#endif
				}

				if (blockDim.x > 32)
				{
					if (lane_id == 0)
						arr_sh[thread_id >> 5] = upd_acc;
					__syncthreads();
				}

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						upd_acc += arr_sh[i];
					double upd_acc_d = (double)upd_acc;

					int accum_bucket_id = block_id & update_accum_mask;

					atomicAddD(update_accum + accum_bucket_id, upd_acc_d);
				}
			}
		}

		template<bool update_accum_flag>
		__global__ void apply_gradient_with_nesterov_momentum_kernel(
			float * __restrict data,
			float * __restrict gradient,
			float * __restrict previous_upd,
			double * __restrict update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			float momentum_plus1,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int block_id = blockIdx.y * gridDim.x + blockIdx.x;
			int elem_id = blockDim.x * block_id + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float prev_upd = __load_nc(previous_upd + elem_id);
				float new_upd = prev_upd * momentum + learning_rate * (gr * normalizer - current_weight * weight_decay);
				float upd = momentum_plus1 * new_upd - momentum * prev_upd;
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				previous_upd[elem_id] = new_upd;
				upd_acc = fabs(upd);
			}

			if (update_accum_flag)
			{
				int thread_id = threadIdx.x;
				int lane_id = thread_id & 31;
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
					#if __CUDACC_VER_MAJOR__ < 9
					upd_acc += __shfl_down(upd_acc, tx);
					#else
					upd_acc += __shfl_down_sync(0xFFFFFFFF, upd_acc, tx);
					#endif
				}

				if (blockDim.x > 32)
				{
					if (lane_id == 0)
						arr_sh[thread_id >> 5] = upd_acc;
					__syncthreads();
				}

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						upd_acc += arr_sh[i];
					double upd_acc_d = (double)upd_acc;

					int accum_bucket_id = block_id & update_accum_mask;

					atomicAddD(update_accum + accum_bucket_id, upd_acc_d);
				}
			}
		}

		template<bool update_accum_flag>
		__global__ void apply_gradient_with_adam_momentum_kernel(
			float * __restrict data,
			float * __restrict gradient,
			float * __restrict biased_first_momentum,
			float * __restrict biased_second_momentum,
			double * __restrict update_accum,
			float alpha,
			float normalizer,
			float weight_decay,
			float beta1,
			float beta2,
			float one_minus_beta1t_inverted,
			float one_minus_beta2t_inverted,
			float epsilon,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int block_id = blockIdx.y * gridDim.x + blockIdx.x;
			int elem_id = blockDim.x * block_id + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float previous_biased_first_momentum = __load_nc(biased_first_momentum + elem_id);
				float previous_biased_second_momentum = __load_nc(biased_second_momentum + elem_id);
				float total_gradient = gr * normalizer - current_weight * weight_decay;
				float new_biased_first_momentum = beta1 * previous_biased_first_momentum + (1.0F - beta1) * total_gradient;
				float new_biased_second_momentum = beta2 * previous_biased_second_momentum + (1.0F - beta2) * total_gradient * total_gradient;
				float unbiased_first_momentum = new_biased_first_momentum * one_minus_beta1t_inverted;
				float unbiased_second_momentum = new_biased_second_momentum * one_minus_beta2t_inverted;
				float upd = __fdividef(alpha * unbiased_first_momentum, sqrtf(unbiased_second_momentum) + epsilon);
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				biased_first_momentum[elem_id] = new_biased_first_momentum;
				biased_second_momentum[elem_id] = new_biased_second_momentum;
				upd_acc = fabs(upd);
			}

			if (update_accum_flag)
			{
				int thread_id = threadIdx.x;
				int lane_id = thread_id & 31;
				#pragma unroll
				for(int tx = 16; tx > 0; tx >>= 1)
				{
					#if __CUDACC_VER_MAJOR__ < 9
					upd_acc += __shfl_down(upd_acc, tx);
					#else
					upd_acc += __shfl_down_sync(0xFFFFFFFF, upd_acc, tx);
					#endif
				}

				if (blockDim.x > 32)
				{
					if (lane_id == 0)
						arr_sh[thread_id >> 5] = upd_acc;
					__syncthreads();
				}

				if (thread_id == 0)
				{
					for(int i = 1; i < (blockDim.x >> 5); ++i)
						upd_acc += arr_sh[i];
					double upd_acc_d = (double)upd_acc;

					int accum_bucket_id = block_id & update_accum_mask;

					atomicAddD(update_accum + accum_bucket_id, upd_acc_d);
				}
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
				if (threadblock_size_x_evenly_divisible <= static_cast<unsigned int>(cuda_config.max_threads_dim[0]))
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
			if (threadblocks <= static_cast<unsigned int>(cuda_config.max_grid_size[0]))
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
			float * output_buf_with_aligned_size,
			const float * input_buf_with_aligned_size,
			float v,
			int elem_count,
			bool add_to_destination,
			cudaStream_t cuda_stream)
		{
			int new_elem_count = (elem_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				new_elem_count);
			if (add_to_destination)
				multiply_by_value_util_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((float4 *)input_buf_with_aligned_size, (float4 *)output_buf_with_aligned_size, v, new_elem_count);
			else
			{
				if ((v != 1.0F) || (output_buf_with_aligned_size != input_buf_with_aligned_size))
				{
					multiply_by_value_util_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>((float4 *)input_buf_with_aligned_size, (float4 *)output_buf_with_aligned_size, v, new_elem_count);
				}
			}
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

		void cuda_util::transpose(
			const cuda_running_configuration& cuda_config,
			const float * src,
			float * dst,
			int src_fast_dim,
			int src_slow_dim,
			int entry_count,
			cudaStream_t cuda_stream)
		{
			dim3 threadblock_size(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
			dim3 grid_size((src_fast_dim + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, (src_slow_dim + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, entry_count);

			transpose_kernel<<<grid_size, threadblock_size, 0, cuda_stream>>>(
				src,
				dst,
				src_fast_dim,
				src_slow_dim,
				src_fast_dim * src_slow_dim,
				entry_count);
		}

		void cuda_util::transpose23(
			const cuda_running_configuration& cuda_config,
			const float * src,
			float * dst,
			int src_dim1,
			int src_dim2,
			int src_dim3,
			cudaStream_t cuda_stream,
			bool add_to_destination)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				src_dim1,
				src_dim2,
				src_dim3);
			if (add_to_destination)
				transpose23_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					src,
					dst,
					src_dim1,
					src_dim2,
					src_dim3);
			else
				transpose23_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					src,
					dst,
					src_dim1,
					src_dim2,
					src_dim3);
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

		void cuda_util::duplicate_vector(
			const cuda_running_configuration& cuda_config,
			const float * source_buf,
			float * dest_buf,
			unsigned int vector_elem_count,
			unsigned int dup_count,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				vector_elem_count,
				(dup_count + 4 - 1) / 4,
				1);
			duplicate_vector_kernel<<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
				source_buf,
				dest_buf,
				vector_elem_count,
				dup_count);
		}

		void cuda_util::apply_gradient(
			const cuda_running_configuration& cuda_config,
			float * data,
			float * gradient,
			double * update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			int elem_count,
			unsigned int update_accum_mask,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count,
				1,
				1,
				32);
			if (update_accum)
			{
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				apply_gradient_kernel<true><<<kernel_dims.first, kernel_dims.second, smem_size, cuda_stream>>>(
					data,
					gradient,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					elem_count,
					update_accum_mask);
			}
			else
				apply_gradient_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					data,
					gradient,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					elem_count,
					update_accum_mask);
		}

		void cuda_util::apply_gradient_with_vanilla_momentum(
			const cuda_running_configuration& cuda_config,
			float * data,
			float * gradient,
			float * prev_upd,
			double * update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			int elem_count,
			unsigned int update_accum_mask,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count,
				1,
				1,
				32);
			if (update_accum)
			{
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				apply_gradient_with_vanilla_momentum_kernel<true><<<kernel_dims.first, kernel_dims.second, smem_size, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					elem_count,
					update_accum_mask);
			}
			else
				apply_gradient_with_vanilla_momentum_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					elem_count,
					update_accum_mask);
		}

		void cuda_util::apply_gradient_with_nesterov_momentum(
			const cuda_running_configuration& cuda_config,
			float * data,
			float * gradient,
			float * prev_upd,
			double * update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			int elem_count,
			unsigned int update_accum_mask,
			cudaStream_t cuda_stream)
		{
			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count,
				1,
				1,
				32);
			if (update_accum)
			{
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				apply_gradient_with_nesterov_momentum_kernel<true><<<kernel_dims.first, kernel_dims.second, smem_size, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					momentum + 1.0F,
					elem_count,
					update_accum_mask);
			}
			else
				apply_gradient_with_nesterov_momentum_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					momentum + 1.0F,
					elem_count,
					update_accum_mask);
		}

		void cuda_util::apply_gradient_with_adam_momentum(
			const cuda_running_configuration& cuda_config,
			float * data,
			float * gradient,
			float * prev_upd,
			float * prev_upd2,
			double * update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			float momentum2,
			int elem_count,
			unsigned int update_accum_mask,
			unsigned int iteration_id,
			cudaStream_t cuda_stream)
		{
			iteration_id = std::max(iteration_id, 1U);

			std::pair<dim3, dim3> kernel_dims = get_grid_and_threadblock_sizes_sequential_access(
				cuda_config,
				elem_count,
				1,
				1,
				32);
			if (update_accum)
			{
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				apply_gradient_with_adam_momentum_kernel<true><<<kernel_dims.first, kernel_dims.second, smem_size, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					prev_upd2,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					momentum2,
					1.0F / (1.0F - powf(momentum, static_cast<float>(iteration_id))),
					1.0F / (1.0F - powf(momentum2, static_cast<float>(iteration_id))),
					1.0e-8F,
					elem_count,
					update_accum_mask);
			}
			else
				apply_gradient_with_adam_momentum_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, cuda_stream>>>(
					data,
					gradient,
					prev_upd,
					prev_upd2,
					update_accum,
					learning_rate,
					normalizer,
					weight_decay,
					momentum,
					momentum2,
					1.0F / (1.0F - powf(momentum, static_cast<float>(iteration_id))),
					1.0F / (1.0F - powf(momentum2, static_cast<float>(iteration_id))),
					1.0e-8F,
					elem_count,
					update_accum_mask);
		}

		void cuda_util::dump_list(
			const float * buffer,
			size_t elem_count,
			const char * filepath,
			cudaStream_t cuda_stream,
			unsigned int elem_count_per_line)
		{
			std::vector<float> elems(elem_count);
			cuda_safe_call(cudaMemcpyAsync(&elems[0], buffer, elem_count * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
			cuda_safe_call(cudaStreamSynchronize(cuda_stream));
			debug_util::dump_list(&elems[0], elem_count, filepath, elem_count_per_line);
		}

		void cuda_util::dump_list(
			const int * buffer,
			size_t elem_count,
			const char * filepath,
			cudaStream_t cuda_stream,
			unsigned int elem_count_per_line)
		{
			std::vector<int> elems(elem_count);
			cuda_safe_call(cudaMemcpyAsync(&elems[0], buffer, elem_count * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
			cuda_safe_call(cudaStreamSynchronize(cuda_stream));
			debug_util::dump_list(&elems[0], elem_count, filepath, elem_count_per_line);
		}
	}
}
