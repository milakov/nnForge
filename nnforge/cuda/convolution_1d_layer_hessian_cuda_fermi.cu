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

#include "convolution_1d_layer_hessian_cuda_fermi.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../convolution_layer.h"
#include "../nn_types.h"

texture<float, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;
texture<float, cudaTextureType1D, cudaReadModeElementType> output_tex_ref;
texture<float, cudaTextureType1D, cudaReadModeElementType> input_squared_tex_ref;

#define FEATURE_MAP_BLOCK_SIZE 4
#define WINDOW_WIDTH_LOCAL 4

namespace nnforge
{
	namespace cuda
	{
		template<int BLOCK_SIZE>
		__global__ void convolution_1d_tex_blocked_hess_kernel_fermi(
			float * __restrict output,
			const float * __restrict weights,
			const float * __restrict biases,
			int output_width,
			int input_width,
			int window_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < output_width) && (output_feature_map_id < output_feature_map_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = window_width * input_feature_map_count;
				int input_elem_id = entry_id * input_feature_map_count * input_width + x;
				const float * current_weights = weights + (int)(weight_count_per_output_feature_map * output_feature_map_id);

				float bias_list[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					if (i < output_feature_map_count - output_feature_map_id)
						bias_list[i] = biases[output_feature_map_id + i];
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = bias_list[i];
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count; ++input_layer_id)
				{
					#pragma unroll 4
					for(int input_x = 0; input_x < window_width; ++input_x)
					{
						float weight_list[FEATURE_MAP_BLOCK_SIZE];
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							weight_list[i] = current_weights[weight_offsets[i]];
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							float inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
						}
						current_weights++;
						input_elem_id++;
					}
					input_elem_id += input_width - window_width;
				}

				float * base_output = output + (entry_id * output_feature_map_count + output_feature_map_id) * output_width + x;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_width - x)
								base_output[j + output_width * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, int BLOCK_SIZE>
		__global__ void convolution_1d_tex_exact_blocked_hess_kernel_fermi(
			float * __restrict output,
			const float * __restrict weights,
			const float * __restrict biases,
			int output_width,
			int input_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < output_width) && (output_feature_map_id < output_feature_map_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = WINDOW_WIDTH * input_feature_map_count;
				int input_elem_id = entry_id * input_feature_map_count * input_width + x;
				const float * current_weights = weights + (int)(weight_count_per_output_feature_map * output_feature_map_id);

				float bias_list[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					if (i < output_feature_map_count - output_feature_map_id)
						bias_list[i] = biases[output_feature_map_id + i];
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = bias_list[i];
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count; ++input_layer_id)
				{
					#pragma unroll
					for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
					{
						float weight_list[FEATURE_MAP_BLOCK_SIZE];
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							weight_list[i] = current_weights[weight_offsets[i]];
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							float inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
						}
						current_weights++;
						input_elem_id++;
					}
					input_elem_id += input_width - WINDOW_WIDTH;
				}

				float * base_output = output + (entry_id * output_feature_map_count + output_feature_map_id) * output_width + x;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_width - x)
								base_output[j + output_width * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		extern __shared__ float arr[];
		__global__ void convolution_1d_update_biases_hess_kernel_fermi(
			float * __restrict hessian_biases,
			const float * __restrict output_errors,
			int block_size,
			int output_elem_count_per_feature_map,
			int output_feature_map_count,
			int entry_count)
		{
			int output_neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = blockIdx.y;
			int block_id = blockIdx.z * blockDim.z + threadIdx.z;
			int base_entry_id = block_size * block_id;
			int thread_id = blockDim.x * threadIdx.z + threadIdx.x;
			int threadblock_size = blockDim.x * blockDim.z;
			float sum = 0.0F;
			int iteration_count = min(entry_count - base_entry_id, block_size);
			if (output_neuron_id < output_elem_count_per_feature_map)
			{
				const float * current_error = output_errors + (base_entry_id * output_feature_map_count + output_feature_map_id) * output_elem_count_per_feature_map + output_neuron_id;
				int output_elem_count_per_entry = output_elem_count_per_feature_map * output_feature_map_count;
				for(int i = 0; i < iteration_count; ++i)
				{
					sum += *current_error;
					current_error += output_elem_count_per_entry;
				}
			}
			arr[thread_id] = sum;
			__syncthreads();

			int t_add_elems = threadblock_size >> 1;
			int t_working_elems = (threadblock_size + 1) >> 1;
			while (t_add_elems > 0)
			{
				if (thread_id < t_add_elems)
					arr[thread_id] += arr[thread_id + t_working_elems];
				t_add_elems = t_working_elems >> 1;
				t_working_elems = (t_working_elems + 1) >> 1;
				__syncthreads();
			}

			if (thread_id == 0)
				atomicAdd(hessian_biases + output_feature_map_id, arr[0]);
		}

		template<int BLOCK_SIZE>
		__global__ void convolution_1d_square_deriviative_tex_hess_kernel_fermi(
			float * __restrict input_errors,
			const float * __restrict weights_squared,
			int output_width,
			int input_width,
			int window_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE + (BLOCK_SIZE - 1);
			int input_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < input_width + (BLOCK_SIZE - 1)) && (input_feature_map_id < input_feature_map_count);
			if (in_bounds)
			{
				int weight_count_per_input_feature_map = window_width;
				int output_elem_id = entry_id * output_feature_map_count * output_width + x;
				const float * current_weights = weights_squared + (int)(window_width * input_feature_map_id);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < input_feature_map_count - input_feature_map_id) ? weight_count_per_input_feature_map * i : 0;

				int min_x_exclusive = x - output_width;
				int max_x_inclusive = x;

				for(int output_layer_id = 0; output_layer_id < output_feature_map_count; ++output_layer_id)
				{
					int input_x = 0;
					#pragma unroll 1
					for(; input_x < (window_width - (WINDOW_WIDTH_LOCAL - 1)); input_x += WINDOW_WIDTH_LOCAL)
					{
						float output_vals[BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1];
						#pragma unroll
						for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1; ++i)
						{
							bool b_fit2 = (i > min_x_exclusive) && (i <= max_x_inclusive);;
							if (b_fit2)
								output_vals[i] = tex1Dfetch(output_tex_ref, output_elem_id - i);
							else
								output_vals[i] = 0.0F;
						}
						output_elem_id -= WINDOW_WIDTH_LOCAL;

						#pragma unroll
						for(int input_x_local = 0; input_x_local < WINDOW_WIDTH_LOCAL; ++input_x_local)
						{
							float weight_list[FEATURE_MAP_BLOCK_SIZE];
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								weight_list[i] = current_weights[weight_offsets[i]];

							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									sums[i * BLOCK_SIZE + j] += output_vals[input_x_local + j] * weight_list[i];
							}
							current_weights++;
						}
					}
					#pragma unroll 1
					for(; input_x < window_width; ++input_x)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							bool b_fit2 = (input_x + j > min_x_exclusive) && (input_x + j <= max_x_inclusive);
							if (b_fit2)
							{
								float inp = tex1Dfetch(output_tex_ref, output_elem_id - j);
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									sums[i * BLOCK_SIZE + j] += inp * current_weights[weight_offsets[i]];
							}
						}
						current_weights++;
						output_elem_id--;
					}

					current_weights += window_width * (input_feature_map_count - 1);
					output_elem_id += window_width + output_width;
				}

				float * base_input = input_errors + (entry_id * input_feature_map_count + input_feature_map_id) * input_width + x;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < input_feature_map_count - input_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j > x - input_width)
								*(base_input + input_width * i - j) = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, int BLOCK_SIZE>
		__global__ void convolution_1d_square_deriviative_tex_exact_hess_kernel_fermi(
			float * __restrict input_errors,
			const float * __restrict weights_squared,
			int output_width,
			int input_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count)
		{
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE + (BLOCK_SIZE - 1);
			int input_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (x < input_width + (BLOCK_SIZE - 1)) && (input_feature_map_id < input_feature_map_count);
			if (in_bounds)
			{
				int weight_count_per_input_feature_map = WINDOW_WIDTH;
				int output_elem_id = entry_id * output_feature_map_count * output_width + x;
				const float * current_weights = weights_squared + (int)(WINDOW_WIDTH * input_feature_map_id);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < input_feature_map_count - input_feature_map_id) ? weight_count_per_input_feature_map * i : 0;

				int min_x_exclusive = x - output_width;
				int max_x_inclusive = x;

				unsigned int mask = 0;
				for(int i = BLOCK_SIZE + WINDOW_WIDTH - 2; i >= 0; --i)
					mask = mask << 1 | (((i > min_x_exclusive) && (i <= max_x_inclusive)) ? 1 : 0);

				for(int output_layer_id = 0; output_layer_id < output_feature_map_count; ++output_layer_id)
				{
					float output_vals[BLOCK_SIZE + WINDOW_WIDTH - 1];
					#pragma unroll
					for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH - 1; ++i)
					{
						bool b_fit2 = (((1 << i) & mask) != 0);
						if (b_fit2)
							output_vals[i] = tex1Dfetch(output_tex_ref, output_elem_id - i);
						else
							output_vals[i] = 0.0F;
					}

					#pragma unroll
					for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
					{
						float weight_list[FEATURE_MAP_BLOCK_SIZE];
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							weight_list[i] = current_weights[weight_offsets[i]];

						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								sums[i * BLOCK_SIZE + j] += output_vals[input_x + j] * weight_list[i];
						}
						current_weights++;
					}
					current_weights += WINDOW_WIDTH * (input_feature_map_count - 1);
					output_elem_id += output_width;
				}

				float * base_input = input_errors + (entry_id * input_feature_map_count + input_feature_map_id) * input_width + x;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < input_feature_map_count - input_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j > x - input_width)
								*(base_input + input_width * i - j) = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		__global__ void convolution_1d_update_weights_hess_kernel_fermi(
			float * __restrict hessian_weights,
			const float * __restrict output_errors,
			int output_width,
			int input_width,
			int window_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int window_x_block_count,
			int block_size)
		{
			int weight_x = (blockIdx.x * blockDim.x + threadIdx.x) * WINDOW_WIDTH_LOCAL;
			int feature_map_pair_id = blockIdx.y * blockDim.y + threadIdx.y;
			int output_feature_map_group_id = feature_map_pair_id / input_feature_map_count;
			int output_feature_map_id = FEATURE_MAP_BLOCK_SIZE * output_feature_map_group_id;
			int base_entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * block_size;

			if ((weight_x < window_width) && (output_feature_map_id < output_feature_map_count) && (base_entry_id < entry_count))
			{
				int output_neuron_count_per_feature_map = output_width;
				int input_feature_map_id = feature_map_pair_id - (output_feature_map_group_id * input_feature_map_count);
				int iteration_count = min(block_size, entry_count - base_entry_id);

				const float * current_output_errors = output_errors + (base_entry_id * output_feature_map_count + output_feature_map_id) * output_width;
				int input_elem_id = (base_entry_id * input_feature_map_count + input_feature_map_id) * input_width + weight_x;

				float sums[FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH_LOCAL];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH_LOCAL; ++i)
					sums[i] = 0.0F;

				int output_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					output_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? output_neuron_count_per_feature_map * i : 0;

				for(int t = 0; t < iteration_count; ++t)
				{
					float input_squared_buf[WINDOW_WIDTH_LOCAL];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH_LOCAL; ++i)
					{
						input_squared_buf[i] = tex1Dfetch(input_squared_tex_ref, input_elem_id);
						++input_elem_id;
					}

					for(int x = 0; x < output_width; ++x)
					{
						float output_error_list[FEATURE_MAP_BLOCK_SIZE];
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							output_error_list[i] = current_output_errors[output_offsets[i]];

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH_LOCAL - 1; ++i)
							input_squared_buf[i] = input_squared_buf[i + 1];
						input_squared_buf[WINDOW_WIDTH_LOCAL - 1] = tex1Dfetch(input_squared_tex_ref, input_elem_id);

						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
								sums[i * WINDOW_WIDTH_LOCAL + j] += output_error_list[i] * input_squared_buf[j];

						current_output_errors++;
						input_elem_id++;
					}
					current_output_errors += (output_feature_map_count - 1) * output_width;
					input_elem_id += (input_feature_map_count - 1) * input_width + (window_width - WINDOW_WIDTH_LOCAL);
				}

				float * base_weights = hessian_weights + (output_feature_map_id * input_feature_map_count + input_feature_map_id) * window_width + weight_x;
				int weight_count_per_output_feature_map = input_feature_map_count * window_width;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
							if (j < window_width - weight_x)
								atomicAdd(base_weights + i * weight_count_per_output_feature_map + j, sums[i * WINDOW_WIDTH_LOCAL + j]);
					}
				}
			}
		}

		template<int WINDOW_WIDTH>
		__global__ void convolution_1d_update_weights_exact_hess_kernel_fermi(
			float * __restrict hessian_weights,
			const float * __restrict output_errors,
			int output_width,
			int input_width,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int block_size)
		{
			int input_feature_map_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = (blockIdx.y * blockDim.y + threadIdx.y) * FEATURE_MAP_BLOCK_SIZE;
			int base_entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * block_size;

			if ((input_feature_map_id < input_feature_map_count) && (output_feature_map_id < output_feature_map_count) && (base_entry_id < entry_count))
			{
				int output_neuron_count_per_feature_map = output_width;
				int iteration_count = min(block_size, entry_count - base_entry_id);

				const float * current_output_errors = output_errors + (base_entry_id * output_feature_map_count + output_feature_map_id) * output_width;
				int input_elem_id = (base_entry_id * input_feature_map_count + input_feature_map_id) * input_width;

				float sums[FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH; ++i)
					sums[i] = 0.0F;

				int output_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					output_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? output_neuron_count_per_feature_map * i : 0;

				for(int t = 0; t < iteration_count; ++t)
				{
					float input_squared_buf[WINDOW_WIDTH];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH; ++i)
					{
						input_squared_buf[i] = tex1Dfetch(input_squared_tex_ref, input_elem_id);
						++input_elem_id;
					}

					for(int x = 0; x < output_width; ++x)
					{
						float output_error_list[FEATURE_MAP_BLOCK_SIZE];
						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							output_error_list[i] = current_output_errors[output_offsets[i]];

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH - 1; ++i)
							input_squared_buf[i] = input_squared_buf[i + 1];
						input_squared_buf[WINDOW_WIDTH - 1] = tex1Dfetch(input_squared_tex_ref, input_elem_id);

						#pragma unroll
						for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH; ++j)
								sums[i * WINDOW_WIDTH + j] += output_error_list[i] * input_squared_buf[j];

						current_output_errors++;
						input_elem_id++;
					}
					current_output_errors += (output_feature_map_count - 1) * output_width;
					input_elem_id += (input_feature_map_count - 1) * input_width;
				}

				float * base_weights = hessian_weights + (output_feature_map_id * input_feature_map_count + input_feature_map_id) * WINDOW_WIDTH;
				int weight_count_per_output_feature_map = input_feature_map_count * WINDOW_WIDTH;
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH; ++j)
							atomicAdd(base_weights + i * weight_count_per_output_feature_map + j, sums[i * WINDOW_WIDTH + j]);
					}
				}
			}
		}

		convolution_1d_layer_hessian_cuda_fermi::convolution_1d_layer_hessian_cuda_fermi()
		{
			input_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_tex_ref.normalized = false;
			output_tex_ref.addressMode[0] = cudaAddressModeBorder;
			output_tex_ref.normalized = false;
			input_squared_tex_ref.addressMode[0] = cudaAddressModeBorder;
			input_squared_tex_ref.normalized = false;
		}

		convolution_1d_layer_hessian_cuda_fermi::~convolution_1d_layer_hessian_cuda_fermi()
		{
		}

#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

#define launch_exact_kernel_const_const(window_width_const, block_size_const) \
	convolution_1d_tex_exact_blocked_hess_kernel_fermi<window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, *data[0], *data[1], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

#define launch_exact_kernel_const(window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const(1, block_size_const); \
			break; \
		case 2: \
			launch_exact_kernel_const_const(2, block_size_const); \
			break; \
		case 3: \
			launch_exact_kernel_const_const(3, block_size_const); \
			break; \
		case 4: \
			launch_exact_kernel_const_const(4, block_size_const); \
			break; \
		case 5: \
			launch_exact_kernel_const_const(5, block_size_const); \
			break; \
		case 6: \
			launch_exact_kernel_const_const(6, block_size_const); \
			break; \
		case 7: \
			launch_exact_kernel_const_const(7, block_size_const); \
			break; \
		case 8: \
			launch_exact_kernel_const_const(8, block_size_const); \
			break; \
		case 9: \
			launch_exact_kernel_const_const(9, block_size_const); \
			break; \
		case 10: \
			launch_exact_kernel_const_const(10, block_size_const); \
			break; \
		};

#define launch_exact_kernel(window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_exact_kernel_const(window_width, 1); \
			break; \
		case 2: \
			launch_exact_kernel_const(window_width, 2); \
			break; \
		case 3: \
			launch_exact_kernel_const(window_width, 3); \
			break; \
		case 4: \
			launch_exact_kernel_const(window_width, 4); \
			break; \
		case 5: \
			launch_exact_kernel_const(window_width, 5); \
			break; \
		};

#define launch_kernel_const(block_size_const) \
	convolution_1d_tex_blocked_hess_kernel_fermi<block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, *data[0], *data[1], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], window_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

#define launch_kernel(block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_kernel_const(1); \
			break; \
		case 2: \
			launch_kernel_const(2); \
			break; \
		case 3: \
			launch_kernel_const(3); \
			break; \
		case 4: \
			launch_kernel_const(4); \
			break; \
		case 5: \
			launch_kernel_const(5); \
			break; \
		};

#define launch_backprop_exact_kernel_const_const(window_width_const, block_size_const) \
	convolution_1d_square_deriviative_tex_exact_hess_kernel_fermi<window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *data_squared[0], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

#define launch_backprop_exact_kernel_const(window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const_const(1, block_size_const); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const_const(2, block_size_const); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const_const(3, block_size_const); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const_const(4, block_size_const); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const_const(5, block_size_const); \
			break; \
		case 6: \
			launch_backprop_exact_kernel_const_const(6, block_size_const); \
			break; \
		case 7: \
			launch_backprop_exact_kernel_const_const(7, block_size_const); \
			break; \
		case 8: \
			launch_backprop_exact_kernel_const_const(8, block_size_const); \
			break; \
		case 9: \
			launch_backprop_exact_kernel_const_const(9, block_size_const); \
			break; \
		case 10: \
			launch_backprop_exact_kernel_const_const(10, block_size_const); \
			break; \
		};

#define launch_backprop_exact_kernel(window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const(window_width, 1); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const(window_width, 2); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const(window_width, 3); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const(window_width, 4); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const(window_width, 5); \
			break; \
		};

#define launch_backprop_kernel_const(block_size_const) \
	convolution_1d_square_deriviative_tex_hess_kernel_fermi<block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *data_squared[0], output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], window_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count);

#define launch_backprop_kernel(block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_kernel_const(1); \
			break; \
		case 2: \
			launch_backprop_kernel_const(2); \
			break; \
		case 3: \
			launch_backprop_kernel_const(3); \
			break; \
		case 4: \
			launch_backprop_kernel_const(4); \
			break; \
		case 5: \
			launch_backprop_kernel_const(5); \
			break; \
		};

#define launch_update_weights_exact_kernel_const(window_width_const) \
	convolution_1d_update_weights_exact_hess_kernel_fermi<window_width_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*hessian_data[0], *output_errors_buffer, output_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[0], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count, block_size);

#define launch_update_weights_exact_kernel(window_width) \
	switch (window_width) \
		{ \
		case 1: \
			launch_update_weights_exact_kernel_const(1); \
			break; \
		case 2: \
			launch_update_weights_exact_kernel_const(2); \
			break; \
		case 3: \
			launch_update_weights_exact_kernel_const(3); \
			break; \
		case 4: \
			launch_update_weights_exact_kernel_const(4); \
			break; \
		case 5: \
			launch_update_weights_exact_kernel_const(5); \
			break; \
		case 6: \
			launch_update_weights_exact_kernel_const(6); \
			break; \
		case 7: \
			launch_update_weights_exact_kernel_const(7); \
			break; \
		case 8: \
			launch_update_weights_exact_kernel_const(8); \
			break; \
		case 9: \
			launch_update_weights_exact_kernel_const(9); \
			break; \
		case 10: \
			launch_update_weights_exact_kernel_const(10); \
			break; \
		};

		void convolution_1d_layer_hessian_cuda_fermi::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_tex_ref, *input_neurons_buffer, desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			int block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				(output_configuration_specific.dimension_sizes[0] + block_size - 1) / block_size,
				((output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE),
				entry_count);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				launch_exact_kernel(window_sizes[0], block_size);
			}
			else
			{
				launch_kernel(block_size);
			}
		}

		void convolution_1d_layer_hessian_cuda_fermi::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_squared,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, output_tex_ref, *output_errors_buffer, desc, output_elem_count_per_entry * entry_count * sizeof(float)));

			int block_size = get_block_size(input_configuration_specific.dimension_sizes[0]);
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_2d_access(
				*cuda_config,
				(input_configuration_specific.dimension_sizes[0] + block_size - 1) / block_size,
				((input_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE),
				entry_count);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				launch_backprop_exact_kernel(window_sizes[0], block_size);
			}
			else
			{
				launch_backprop_kernel(block_size);
			}
		}

		void convolution_1d_layer_hessian_cuda_fermi::enqueue_update_hessian(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& hessian_data,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cuda_safe_call(cudaBindTexture(0, input_squared_tex_ref, *additional_buffers[0], desc, input_elem_count_per_entry * entry_count * sizeof(float)));

			// Update weights
			{
				// Store input neurons multiplied element-wise by themselves
				cuda_util::multiply_by_itself(
					*cuda_config,
					*input_neurons_buffer,
					*additional_buffers[0],
					input_elem_count_per_entry * entry_count,
					stream_id);

				if (window_sizes[0] <= MAX_WINDOW_WIDTH)
				{
					int block_size = get_weights_update_block_size(entry_count);
					int block_count = (entry_count + block_size - 1) / block_size;
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						((output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE),
						input_configuration_specific.feature_map_count,
						block_count);

					launch_update_weights_exact_kernel(window_sizes[0]);
				}
				else
				{
					int window_x_block_count = (window_sizes[0] + WINDOW_WIDTH_LOCAL - 1) / WINDOW_WIDTH_LOCAL;
					int block_size = get_weights_update_block_size(entry_count);
					int block_count = (entry_count + block_size - 1) / block_size;
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						window_x_block_count,
						((output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE) * input_configuration_specific.feature_map_count,
						block_count);

					convolution_1d_update_weights_hess_kernel_fermi<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*hessian_data[0],
						*output_errors_buffer,
						output_configuration_specific.dimension_sizes[0],
						input_configuration_specific.dimension_sizes[0],
						window_sizes[0],
						input_configuration_specific.feature_map_count,
						output_configuration_specific.feature_map_count,
						entry_count,
						window_x_block_count,
						block_size);
				}
			}

			// Update biases
			{
				int block_size = get_bias_update_block_size(entry_count);
				int block_count = (entry_count + block_size - 1) / block_size;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					output_elem_count_per_feature_map,
					1,
					block_count);
				kernel_dims.first.y = output_configuration_specific.feature_map_count;
				int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
				int smem_size = threadblock_size * sizeof(float);
				convolution_1d_update_biases_hess_kernel_fermi<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
					*hessian_data[1],
					*output_errors_buffer,
					block_size,
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count);
			}
		}

		int convolution_1d_layer_hessian_cuda_fermi::get_block_size(int width)
		{
			int block_count = (width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
			int block_size = (width + block_count - 1) / block_count;
			return block_size;
		}

		void convolution_1d_layer_hessian_cuda_fermi::hessian_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_sizes.push_back(static_cast<int>(*it));
		}

		bool convolution_1d_layer_hessian_cuda_fermi::is_in_place_backprop() const
		{
			return false;
		}

		std::vector<size_t> convolution_1d_layer_hessian_cuda_fermi::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_entry * sizeof(float));

			return res;
		}

		std::vector<unsigned int> convolution_1d_layer_hessian_cuda_fermi::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);
			res.push_back(output_elem_count_per_entry);

			return res;
		}

		int convolution_1d_layer_hessian_cuda_fermi::get_bias_update_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}

		int convolution_1d_layer_hessian_cuda_fermi::get_weights_update_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
