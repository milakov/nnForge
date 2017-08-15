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

#include "../layer_configuration_specific.h"

static __forceinline__ __device__ float __load_nc(const float * ptr)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr;
#endif
#endif
}

static __forceinline__ __device__ int __load_nc(const int * ptr)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr;
#endif
#endif
}

static __forceinline__ __device__ float2 __load_nc(const float2 * ptr)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr;
#endif
#endif
}

template<typename element_type, int length>
class array_by_val
{
public:
	element_type vals[length];

	__forceinline__ __host__ __device__ const element_type& operator[] (int index) const
	{
		return vals[index];
	}

	__forceinline__ __host__ __device__ element_type& operator[] (int index)
	{
		return vals[index];
	}
};

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

			static void set_with_value(
				const cuda_running_configuration& cuda_config,
				double * buf_with_aligned_size,
				double v,
				int elem_count,
				cudaStream_t cuda_stream);

			static void set_with_value(
				const cuda_running_configuration& cuda_config,
				int * buf_with_aligned_size,
				int v,
				int elem_count,
				cudaStream_t cuda_stream);

			static void multiply_by_value(
				const cuda_running_configuration& cuda_config,
				float * output_buf_with_aligned_size,
				const float * input_buf_with_aligned_size,
				float v,
				int elem_count,
				bool add_to_destination,
				cudaStream_t cuda_stream);

			static void apply_weight_decay(
				const cuda_running_configuration& cuda_config,
				const float * learning_rates_with_aligned_size,
				float * weights_with_aligned_size,
				float weight_decay,
				int elem_count,
				cudaStream_t cuda_stream);

			static void apply_gradient_with_weight_decay(
				const cuda_running_configuration& cuda_config,
				const float * gradient_with_aligned_size,
				const float * learning_rates_with_aligned_size,
				float * weights_with_aligned_size,
				float weight_decay,
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

			static void transpose(
				const cuda_running_configuration& cuda_config,
				const float * src,
				float * dst,
				int src_fast_dim,
				int src_slow_dim,
				int entry_count,
				cudaStream_t cuda_stream);

			static void transpose23(
				const cuda_running_configuration& cuda_config,
				const float * src,
				float * dst,
				int src_dim1,
				int src_dim2,
				int src_dim3,
				cudaStream_t cuda_stream,
				bool add_to_destination = false);

			static int get_group_count(
				const cuda_running_configuration& cuda_config,
				int total_thread_count,
				int divisible);

			static int get_thread_count_per_wave(const cuda_running_configuration& cuda_config);

			static unsigned int get_feature_map_count_striped(unsigned int feature_map_count);

			static layer_configuration_specific get_layer_configuration_specific_striped(const layer_configuration_specific& original_layer_config);

			static void copy_to_striped(
				const cuda_running_configuration& cuda_config,
				const float * source_buf,
				float2 * dest_buf,
				unsigned int elem_count_per_feature_map,
				unsigned int feature_map_count,
				unsigned int entry_count,
				cudaStream_t cuda_stream);

			static void copy_from_striped(
				const cuda_running_configuration& cuda_config,
				const float2 * source_buf,
				float * dest_buf,
				unsigned int elem_count_per_feature_map,
				unsigned int feature_map_count,
				unsigned int entry_count,
				cudaStream_t cuda_stream);

			static void duplicate_vector(
				const cuda_running_configuration& cuda_config,
				const float * source_buf,
				float * dest_buf,
				unsigned int vector_elem_count,
				unsigned int dup_count,
				cudaStream_t cuda_stream);

			static void apply_gradient(
				const cuda_running_configuration& cuda_config,
				float * data,
				float * gradient,
				double * update_accum,
				float learning_rate,
				float normalizer,
				float weight_decay,
				int elem_count,
				unsigned int update_accum_mask,
				cudaStream_t cuda_stream);

			static void apply_gradient_with_vanilla_momentum(
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
				cudaStream_t cuda_stream);

			static void apply_gradient_with_nesterov_momentum(
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
				cudaStream_t cuda_stream);

			static void apply_gradient_with_adam_momentum(
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
				cudaStream_t cuda_stream);

			static void dump_list(
				const float * buffer,
				size_t elem_count,
				const char * filepath,
				cudaStream_t cuda_stream,
				unsigned int elem_count_per_line = 1);

			static void dump_list(
				const int * buffer,
				size_t elem_count,
				const char * filepath,
				cudaStream_t cuda_stream,
				unsigned int elem_count_per_line = 1);

		private:
			cuda_util();
			cuda_util(const cuda_util&);
			cuda_util& operator =(const cuda_util&);
			~cuda_util();

			static const unsigned int preferred_width_2d_access;
			static const unsigned int preferred_height_2d_access;
			static const unsigned int preferred_threadblocksize_sequential_access;
			static const unsigned int preferred_width_2d_access_x_aligned;
			static const unsigned int preferred_height_2d_access_x_aligned;
		};
	}
}
