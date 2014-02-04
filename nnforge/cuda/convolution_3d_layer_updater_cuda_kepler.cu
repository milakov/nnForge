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

#include "convolution_3d_layer_updater_cuda_kepler.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "cuda_texture.h"
#include "packed_config.h"
#include "space_filling_curve.h"

#include "../convolution_layer.h"

#define FEATURE_MAP_BLOCK_SIZE 4
#define WINDOW_WIDTH_LOCAL 4

namespace nnforge
{
	namespace cuda
	{
		template<int BLOCK_SIZE, bool single_input_feature_map_group>
		__global__ void convolution_3d_tex_upd_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_width,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_feature_map_group_size,
			int texture_offset,
			int entry_count,
			bool different_input,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int x = conf.get_val(0);
				int y = conf.get_val(1);
				int z = conf.get_val(2);
				int output_feature_map_id = conf.get_val(3);
				int base_input_feature_map_id = conf.get_val(4);

				int weight_count_per_output_feature_map = window_depth * window_height * window_width * input_feature_map_count;
				int input_elem_id = ((((different_input ? entry_id * input_feature_map_count : 0) + base_input_feature_map_id) * input_depth + z) * input_height + y) * input_width + x + texture_offset;
				int weights_offset = ((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + base_input_feature_map_id) * window_depth * window_height * window_width;
				int iteration_count = min(input_feature_map_group_size, input_feature_map_count - base_input_feature_map_id);

				float initial_values[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					initial_values[i] = 0.0F;
				if (base_input_feature_map_id == 0)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						if (i < output_feature_map_count - output_feature_map_id)
							initial_values[i] = biases[entry_id * output_feature_map_count + output_feature_map_id + i];
				}
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = initial_values[i];

				for(int i = 0; i < iteration_count; ++i)
				{
					for(int input_z = 0; input_z < window_depth; ++input_z)
					{
						for(int input_y = 0; input_y < window_height; ++input_y)
						{
							#pragma unroll 4
							for(int input_x = 0; input_x < window_width; ++input_x)
							{
								float weight_list[FEATURE_MAP_BLOCK_SIZE];
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									weight_list[i] = tex1Dfetch<float>(weights_tex, weights_offset + weight_count_per_output_feature_map * i);
								#pragma unroll
								for(int j = 0; j < BLOCK_SIZE; ++j)
								{
									float inp = tex1Dfetch<float>(input_tex, input_elem_id + j); 
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
								}
								weights_offset++;
								input_elem_id++;
							} // for input_x
							input_elem_id += input_width - window_width;
						} // for input_y
						input_elem_id += input_width * (input_height - window_height);
					} // for input_z
					input_elem_id += input_height * input_width * (input_depth - window_depth);
				}

				float * base_output = output + (((entry_id * output_feature_map_count + output_feature_map_id) * output_depth + z) * output_height + y) * output_width + x;
				int output_neuron_count_per_feature_map = output_depth * output_height * output_width;
				if (single_input_feature_map_group)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j < output_width - x)
									base_output[output_neuron_count_per_feature_map * i + j] = sums[i * BLOCK_SIZE + j];
							}
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j < output_width - x)
									atomicAdd(base_output + output_neuron_count_per_feature_map * i + j, sums[i * BLOCK_SIZE + j]);
							}
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, int BLOCK_SIZE, bool single_input_feature_map_group>
		__global__ void convolution_3d_tex_exact_upd_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_feature_map_group_size,
			int texture_offset,
			int entry_count,
			bool different_input,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int x = conf.get_val(0);
				int y = conf.get_val(1);
				int z = conf.get_val(2);
				int output_feature_map_id = conf.get_val(3);
				int base_input_feature_map_id = conf.get_val(4);

				int weight_count_per_output_feature_map = window_depth * window_height * WINDOW_WIDTH * input_feature_map_count;
				int input_elem_id = ((((different_input ? entry_id * input_feature_map_count : 0) + base_input_feature_map_id) * input_depth + z) * input_height + y) * input_width + x + texture_offset;
				int weights_offset = ((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + base_input_feature_map_id) * window_depth * window_height * WINDOW_WIDTH;
				int iteration_count = min(input_feature_map_group_size, input_feature_map_count - base_input_feature_map_id);

				float initial_values[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					initial_values[i] = 0.0F;
				if (base_input_feature_map_id == 0)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
						if (i < output_feature_map_count - output_feature_map_id)
							initial_values[i] = biases[entry_id * output_feature_map_count + output_feature_map_id + i];
				}
				float sums[BLOCK_SIZE * FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_SIZE; ++j)
						sums[i * BLOCK_SIZE + j] = initial_values[i];

				for(int i = 0; i < iteration_count; ++i)
				{
					for(int input_z = 0; input_z < window_depth; ++input_z)
					{
						for(int input_y = 0; input_y < window_height; ++input_y)
						{
							#pragma unroll
							for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
							{
								float weight_list[FEATURE_MAP_BLOCK_SIZE];
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									weight_list[i] = tex1Dfetch<float>(weights_tex, weights_offset + weight_count_per_output_feature_map * i);
								#pragma unroll
								for(int j = 0; j < BLOCK_SIZE; ++j)
								{
									float inp = tex1Dfetch<float>(input_tex, input_elem_id + j); 
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										sums[i * BLOCK_SIZE + j] += inp * weight_list[i];
								}
								weights_offset++;
								input_elem_id++;
							} // for input_x
							input_elem_id += input_width - WINDOW_WIDTH;
						} // for input_y
						input_elem_id += input_width * (input_height - window_height);
					} // for input_z
					input_elem_id += input_height * input_width * (input_depth - window_depth);
				}

				float * base_output = output + (((entry_id * output_feature_map_count + output_feature_map_id) * output_depth + z) * output_height + y) * output_width + x;
				int output_neuron_count_per_feature_map = output_depth * output_height * output_width;
				if (single_input_feature_map_group)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j < output_width - x)
									base_output[output_neuron_count_per_feature_map * i + j] = sums[i * BLOCK_SIZE + j];
							}
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j < output_width - x)
									atomicAdd(base_output + output_neuron_count_per_feature_map * i + j, sums[i * BLOCK_SIZE + j]);
							}
						}
					}
				}
			}
		}

		__global__ void convolution_3d_update_biases_upd_kernel_kepler(
			float * __restrict biases,
			const float * __restrict output_errors,
			const float * __restrict learning_rate,
			int output_feature_map_count,
			int output_elem_count_per_feature_map,
			int min_iteration_count)
		{
			int thread_id = threadIdx.x;
			int output_feature_map_id = blockIdx.y;
			int entry_id = blockIdx.z;
			int threadblock_size = blockDim.x;

			float sum = 0.0F;
			const float * current_error = output_errors + (entry_id * output_feature_map_count + output_feature_map_id) * output_elem_count_per_feature_map;
			int current_output_neuron_id = thread_id;
			for(int i = 0; i < min_iteration_count; ++i)
			{
				sum += current_error[current_output_neuron_id];
				current_output_neuron_id += threadblock_size;
			}
			if (current_output_neuron_id < output_elem_count_per_feature_map)
				sum += current_error[current_output_neuron_id];

			int lane_id = thread_id & 31;
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
				sum += __shfl_down(sum, tx);
			}

			if (lane_id == 0)
			{
				int offset = entry_id * output_feature_map_count + output_feature_map_id;
				float current_learning_rate_val = learning_rate[offset];
				atomicAdd(biases + offset, sum * current_learning_rate_val);
			}
		}

		template<int BLOCK_SIZE, bool single_output_feature_map_group>
		__global__ void convolution_3d_deriviative_tex_upd_kernel_kepler(
			float * __restrict input_errors,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t weights_tex,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_width,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_feature_map_group_size,
			int entry_count,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int x = conf.get_val(0);
				int y = conf.get_val(1);
				int z = conf.get_val(2);
				int input_feature_map_id = conf.get_val(3);
				int base_output_feature_map_id = conf.get_val(4);

				int weight_count_per_input_feature_map = window_depth * window_height * window_width;
				int output_elem_id = (((entry_id * output_feature_map_count + base_output_feature_map_id) * output_depth + z) * output_height + y) * output_width + x;
				int weights_offset = ((entry_id * output_feature_map_count + base_output_feature_map_id) * input_feature_map_count + input_feature_map_id) * weight_count_per_input_feature_map;
				int iteration_count = min(output_feature_map_group_size, output_feature_map_count - base_output_feature_map_id);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int min_z_exclusive = z - output_depth;
				int max_z_inclusive = z;
				int min_y_exclusive = y - output_height;
				int max_y_inclusive = y;
				int min_x_exclusive = x - output_width;
				int max_x_inclusive = x;

				for(int i = 0; i < iteration_count; ++i)
				{
					for(int input_z = 0; input_z < window_depth; ++input_z)
					{
						bool b_fit_z = (input_z > min_z_exclusive) && (input_z <= max_z_inclusive);

						for(int input_y = 0; input_y < window_height; ++input_y)
						{
							bool b_fit_y = b_fit_z && (input_y > min_y_exclusive) && (input_y <= max_y_inclusive);

							int input_x = 0;
							#pragma unroll 1
							for(; input_x < (window_width - (WINDOW_WIDTH_LOCAL - 1)); input_x += WINDOW_WIDTH_LOCAL)
							{
								float output_vals[BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1];
								#pragma unroll
								for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1; ++i)
								{
									bool b_fit_x = b_fit_y && (i > min_x_exclusive) && (i <= max_x_inclusive);;
									if (b_fit_x)
										output_vals[i] = tex1Dfetch<float>(output_tex, output_elem_id - i);
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
										weight_list[i] = tex1Dfetch<float>(weights_tex, weights_offset + weight_count_per_input_feature_map * i);

									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											sums[i * BLOCK_SIZE + j] += output_vals[input_x_local + j] * weight_list[i];
									}
									weights_offset++;
								}
							}
							#pragma unroll 1
							for(; input_x < window_width; ++input_x)
							{
								#pragma unroll
								for(int j = 0; j < BLOCK_SIZE; ++j)
								{
									bool b_fit_x = b_fit_y && (input_x + j > min_x_exclusive) && (input_x + j <= max_x_inclusive);
									if (b_fit_x)
									{
										float inp = tex1Dfetch<float>(output_tex, output_elem_id - j);
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											sums[i * BLOCK_SIZE + j] += inp * tex1Dfetch<float>(weights_tex, weights_offset + weight_count_per_input_feature_map * i);
									}
								}
								weights_offset++;
								output_elem_id--;
							}

							output_elem_id += window_width - output_width;
						} // for input_y
						output_elem_id += output_width * (window_height - output_height);
					} // for input_z
					output_elem_id += output_width * output_height * (output_depth + window_depth);
					weights_offset += weight_count_per_input_feature_map * (input_feature_map_count - 1);
				}

				float * base_input = input_errors + (((entry_id * input_feature_map_count + input_feature_map_id) * input_depth + z) * input_height + y) * input_width + x;
				int input_neuron_count_per_feature_map = input_depth * input_height * input_width;
				if (single_output_feature_map_group == 1)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < input_feature_map_count - input_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j > x - input_width)
									*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
							}
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < input_feature_map_count - input_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j > x - input_width)
									atomicAdd(base_input + input_neuron_count_per_feature_map * i - j, sums[i * BLOCK_SIZE + j]);
							}
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, int BLOCK_SIZE, bool single_output_feature_map_group>
		__global__ void convolution_3d_deriviative_tex_exact_upd_kernel_kepler(
			float * __restrict input_errors,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t weights_tex,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_feature_map_group_size,
			int entry_count,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int x = conf.get_val(0);
				int y = conf.get_val(1);
				int z = conf.get_val(2);
				int input_feature_map_id = conf.get_val(3);
				int base_output_feature_map_id = conf.get_val(4);

				int weight_count_per_input_feature_map = window_depth * window_height * WINDOW_WIDTH;
				int output_elem_id = (((entry_id * output_feature_map_count + base_output_feature_map_id) * output_depth + z) * output_height + y) * output_width + x;
				int weights_offset = ((entry_id * output_feature_map_count + base_output_feature_map_id) * input_feature_map_count + input_feature_map_id) * weight_count_per_input_feature_map;
				int iteration_count = min(output_feature_map_group_size, output_feature_map_count - base_output_feature_map_id);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int min_z_exclusive = z - output_depth;
				int max_z_inclusive = z;
				int min_y_exclusive = y - output_height;
				int max_y_inclusive = y;
				int min_x_exclusive = x - output_width;
				int max_x_inclusive = x;

				unsigned int mask = 0;
				for(int i = BLOCK_SIZE + WINDOW_WIDTH - 2; i >= 0; --i)
					mask = mask << 1 | (((i > min_x_exclusive) && (i <= max_x_inclusive)) ? 1 : 0);

				for(int i = 0; i < iteration_count; ++i)
				{
					for(int input_z = 0; input_z < window_depth; ++input_z)
					{
						bool b_fit_z = (input_z > min_z_exclusive) && (input_z <= max_z_inclusive);

						for(int input_y = 0; input_y < window_height; ++input_y)
						{
							bool b_fit_y = b_fit_z && (input_y > min_y_exclusive) && (input_y <= max_y_inclusive);

							float output_vals[BLOCK_SIZE + WINDOW_WIDTH - 1];
							#pragma unroll
							for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH - 1; ++i)
							{
								bool b_fit_x = b_fit_y && (((1 << i) & mask) != 0);
								if (b_fit_x)
									output_vals[i] = tex1Dfetch<float>(output_tex, output_elem_id - i);
								else
									output_vals[i] = 0.0F;
							}

							#pragma unroll
							for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
							{
								float weight_list[FEATURE_MAP_BLOCK_SIZE];
								#pragma unroll
								for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
									weight_list[i] = tex1Dfetch<float>(weights_tex, weights_offset + weight_count_per_input_feature_map * i);

								#pragma unroll
								for(int j = 0; j < BLOCK_SIZE; ++j)
								{
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										sums[i * BLOCK_SIZE + j] += output_vals[input_x + j] * weight_list[i];
								}
								weights_offset++;
							}
							output_elem_id -= output_width;
						} // for input_y
						output_elem_id += output_width * (window_height - output_height);
					} // for input_z
					output_elem_id += output_width * output_height * (output_depth + window_depth);
					weights_offset += weight_count_per_input_feature_map * (input_feature_map_count - 1);
				}

				float * base_input = input_errors + (((entry_id * input_feature_map_count + input_feature_map_id) * input_depth + z) * input_height + y) * input_width + x;
				int input_neuron_count_per_feature_map = input_depth * input_height * input_width;
				if (single_output_feature_map_group == 1)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < input_feature_map_count - input_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j > x - input_width)
									*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
							}
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < input_feature_map_count - input_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < BLOCK_SIZE; ++j)
							{
								if (j > x - input_width)
									atomicAdd(base_input + input_neuron_count_per_feature_map * i - j, sums[i * BLOCK_SIZE + j]);
							}
						}
					}
				}
			}
		}

		template<bool single_output_z_group>
		__global__ void convolution_3d_update_weights_upd_kernel_kepler(
			float * __restrict weights,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t output_tex,
			const float * __restrict learning_rate,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_width,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_z_group_count,
			int texture_offset,
			int entry_count,
			bool different_input,
			int packed_config_count)
		{
			int weight_x = (blockIdx.x * blockDim.x + threadIdx.x) * WINDOW_WIDTH_LOCAL;
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (packed_config_id < packed_config_count) && (entry_id < entry_count) && (weight_x < window_width); 
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int weight_y = conf.get_val(0);
				int weight_z = conf.get_val(1);
				int input_feature_map_id = conf.get_val(2);
				int output_feature_map_id = conf.get_val(3);
				int output_z_start_id = conf.get_val(4);

				int output_neuron_count_per_feature_map = output_depth * output_width * output_height;
				int output_elem_id = (((entry_id * output_feature_map_count + output_feature_map_id) * output_depth + output_z_start_id) * output_height) * output_width;
				int input_elem_id = ((((different_input ? entry_id * input_feature_map_count : 0) + input_feature_map_id) * input_depth + weight_z + output_z_start_id) * input_height + weight_y) * input_width + texture_offset + weight_x;

				float sums[FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH_LOCAL];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH_LOCAL; ++i)
					sums[i] = 0.0F;

				for(int output_z = output_z_start_id; output_z < output_depth; output_z += output_z_group_count)
				{
					for(int output_y = 0; output_y < output_height; output_y++)
					{
						float input_buf[WINDOW_WIDTH_LOCAL];
						#pragma unroll
						for(int i = 1; i < WINDOW_WIDTH_LOCAL; ++i)
						{
							input_buf[i] = tex1Dfetch<float>(input_tex, input_elem_id);
							++input_elem_id;
						}

						for(int x = 0; x < output_width; ++x)
						{
							float output_error_list[FEATURE_MAP_BLOCK_SIZE];
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								output_error_list[i] = tex1Dfetch<float>(output_tex, output_elem_id + output_neuron_count_per_feature_map * i);

							#pragma unroll
							for(int i = 0; i < WINDOW_WIDTH_LOCAL - 1; ++i)
								input_buf[i] = input_buf[i + 1];
							input_buf[WINDOW_WIDTH_LOCAL - 1] = tex1Dfetch<float>(input_tex, input_elem_id);

							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								#pragma unroll
								for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
									sums[i * WINDOW_WIDTH_LOCAL + j] += output_error_list[i] * input_buf[j];

							output_elem_id++;
							input_elem_id++;
						}

						input_elem_id += window_width - WINDOW_WIDTH_LOCAL;
					}

					output_elem_id += output_height * output_width * (output_z_group_count - 1);
					input_elem_id += input_height * input_width * (output_z_group_count - 1) + (input_width * (window_height - 1));
				}

				int offset = ((((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + input_feature_map_id) * window_depth + weight_z) * window_height + weight_y) * window_width + weight_x;
				int weight_count_per_output_feature_map = input_feature_map_count * window_depth * window_height * window_width;
				float * cur_weights = weights + offset;
				const float * cur_learning_rate = learning_rate + offset;
				if (single_output_z_group)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
								if (j < window_width - weight_x)
									cur_weights[i * weight_count_per_output_feature_map + j] += sums[i * WINDOW_WIDTH_LOCAL + j] * cur_learning_rate[i * weight_count_per_output_feature_map + j];
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
								if (j < window_width - weight_x)
									atomicAdd(cur_weights + i * weight_count_per_output_feature_map + j, sums[i * WINDOW_WIDTH_LOCAL + j] * cur_learning_rate[i * weight_count_per_output_feature_map + j]);
						}
					}
				}
			}
		}

		template<int WINDOW_WIDTH, bool single_output_z_group>
		__global__ void convolution_3d_update_weights_exact_upd_kernel_kepler(
			float * __restrict weights,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t output_tex,
			const float * __restrict learning_rate,
			const packed_config<5> * __restrict packed_config_list,
			int output_width,
			int output_height,
			int output_depth,
			int input_width,
			int input_height,
			int input_depth,
			int window_height,
			int window_depth,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_z_group_count,
			int texture_offset,
			int entry_count,
			bool different_input,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (packed_config_id < packed_config_count) && (entry_id < entry_count);
			if (in_bounds)
			{
				packed_config<5> conf = packed_config_list[packed_config_id];
				int weight_y = conf.get_val(0);
				int weight_z = conf.get_val(1);
				int input_feature_map_id = conf.get_val(2);
				int output_feature_map_id = conf.get_val(3);
				int output_z_start_id = conf.get_val(4);

				int output_neuron_count_per_feature_map = output_depth * output_width * output_height;
				int output_elem_id = (((entry_id * output_feature_map_count + output_feature_map_id) * output_depth + output_z_start_id) * output_height) * output_width;
				int input_elem_id = ((((different_input ? entry_id * input_feature_map_count : 0) + input_feature_map_id) * input_depth + weight_z + output_z_start_id) * input_height + weight_y) * input_width + texture_offset;

				float sums[FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * WINDOW_WIDTH; ++i)
					sums[i] = 0.0F;

				for(int output_z = output_z_start_id; output_z < output_depth; output_z += output_z_group_count)
				{
					for(int output_y = 0; output_y < output_height; output_y++)
					{
						float input_buf[WINDOW_WIDTH];
						#pragma unroll
						for(int i = 1; i < WINDOW_WIDTH; ++i)
						{
							input_buf[i] = tex1Dfetch<float>(input_tex, input_elem_id);
							++input_elem_id;
						}

						for(int x = 0; x < output_width; ++x)
						{
							float output_error_list[FEATURE_MAP_BLOCK_SIZE];
							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								output_error_list[i] = tex1Dfetch<float>(output_tex, output_elem_id + output_neuron_count_per_feature_map * i);

							#pragma unroll
							for(int i = 0; i < WINDOW_WIDTH - 1; ++i)
								input_buf[i] = input_buf[i + 1];
							input_buf[WINDOW_WIDTH - 1] = tex1Dfetch<float>(input_tex, input_elem_id);

							#pragma unroll
							for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
								#pragma unroll
								for(int j = 0; j < WINDOW_WIDTH; ++j)
									sums[i * WINDOW_WIDTH + j] += output_error_list[i] * input_buf[j];

							output_elem_id++;
							input_elem_id++;
						}
					}
					output_elem_id += output_height * output_width * (output_z_group_count - 1);
					input_elem_id += input_height * input_width * (output_z_group_count - 1) + (input_width * (window_height - 1));
				}

				int offset = ((((entry_id * output_feature_map_count + output_feature_map_id) * input_feature_map_count + input_feature_map_id) * window_depth + weight_z) * window_height + weight_y) * WINDOW_WIDTH;
				int weight_count_per_output_feature_map = input_feature_map_count * window_depth * window_height * WINDOW_WIDTH;
				float * cur_weights = weights + offset;
				const float * cur_learning_rate = learning_rate + offset;
				if (single_output_z_group)
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH; ++j)
								cur_weights[i * weight_count_per_output_feature_map + j] += sums[i * WINDOW_WIDTH + j] * cur_learning_rate[i * weight_count_per_output_feature_map + j];
						}
					}
				}
				else
				{
					#pragma unroll
					for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					{
						if (i < output_feature_map_count - output_feature_map_id)
						{
							#pragma unroll
							for(int j = 0; j < WINDOW_WIDTH; ++j)
								atomicAdd(cur_weights + i * weight_count_per_output_feature_map + j, sums[i * WINDOW_WIDTH + j] * cur_learning_rate[i * weight_count_per_output_feature_map + j]);
						}
					}
				}
			}
		}

		convolution_3d_layer_updater_cuda_kepler::convolution_3d_layer_updater_cuda_kepler()
		{
		}

		convolution_3d_layer_updater_cuda_kepler::~convolution_3d_layer_updater_cuda_kepler()
		{
		}

#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

#define launch_exact_kernel_const_const(window_width_const, block_size_const, single_input_feature_map_group) \
	convolution_3d_tex_exact_upd_kernel_kepler<window_width_const,block_size_const,single_input_feature_map_group><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, forward_input_feature_map_group_size, texture_offset, entry_count, different_input, packed_config_count);

#define launch_exact_kernel_const(window_width, block_size_const, single_input_feature_map_group) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const(1, block_size_const, single_input_feature_map_group); \
			break; \
		case 2: \
			launch_exact_kernel_const_const(2, block_size_const, single_input_feature_map_group); \
			break; \
		case 3: \
			launch_exact_kernel_const_const(3, block_size_const, single_input_feature_map_group); \
			break; \
		case 4: \
			launch_exact_kernel_const_const(4, block_size_const, single_input_feature_map_group); \
			break; \
		case 5: \
			launch_exact_kernel_const_const(5, block_size_const, single_input_feature_map_group); \
			break; \
		case 6: \
			launch_exact_kernel_const_const(6, block_size_const, single_input_feature_map_group); \
			break; \
		case 7: \
			launch_exact_kernel_const_const(7, block_size_const, single_input_feature_map_group); \
			break; \
		case 8: \
			launch_exact_kernel_const_const(8, block_size_const, single_input_feature_map_group); \
			break; \
		case 9: \
			launch_exact_kernel_const_const(9, block_size_const, single_input_feature_map_group); \
			break; \
		case 10: \
			launch_exact_kernel_const_const(10, block_size_const, single_input_feature_map_group); \
			break; \
		};

#define launch_exact_kernel(window_width, block_size, single_input_feature_map_group) \
	switch (block_size) \
		{ \
		case 1: \
			launch_exact_kernel_const(window_width, 1, single_input_feature_map_group); \
			break; \
		case 2: \
			launch_exact_kernel_const(window_width, 2, single_input_feature_map_group); \
			break; \
		case 3: \
			launch_exact_kernel_const(window_width, 3, single_input_feature_map_group); \
			break; \
		case 4: \
			launch_exact_kernel_const(window_width, 4, single_input_feature_map_group); \
			break; \
		case 5: \
			launch_exact_kernel_const(window_width, 5, single_input_feature_map_group); \
			break; \
		};

#define launch_kernel_const(block_size_const, single_input_feature_map_group) \
	convolution_3d_tex_upd_kernel_kepler<block_size_const,single_input_feature_map_group><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[0], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, forward_input_feature_map_group_size, texture_offset, entry_count, different_input, packed_config_count);

#define launch_kernel(block_size, single_input_feature_map_group) \
	switch (block_size) \
		{ \
		case 1: \
			launch_kernel_const(1, single_input_feature_map_group); \
			break; \
		case 2: \
			launch_kernel_const(2, single_input_feature_map_group); \
			break; \
		case 3: \
			launch_kernel_const(3, single_input_feature_map_group); \
			break; \
		case 4: \
			launch_kernel_const(4, single_input_feature_map_group); \
			break; \
		case 5: \
			launch_kernel_const(5, single_input_feature_map_group); \
			break; \
		};

#define launch_backprop_exact_kernel_const_const(window_width_const, block_size_const, single_output_feature_map_group) \
	convolution_3d_deriviative_tex_exact_upd_kernel_kepler<window_width_const,block_size_const,single_output_feature_map_group><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, weights_tex, packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, backward_output_feature_map_group_size, entry_count, packed_config_count);

#define launch_backprop_exact_kernel_const(window_width, block_size_const, single_output_feature_map_group) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const_const(1, block_size_const, single_output_feature_map_group); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const_const(2, block_size_const, single_output_feature_map_group); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const_const(3, block_size_const, single_output_feature_map_group); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const_const(4, block_size_const, single_output_feature_map_group); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const_const(5, block_size_const, single_output_feature_map_group); \
			break; \
		case 6: \
			launch_backprop_exact_kernel_const_const(6, block_size_const, single_output_feature_map_group); \
			break; \
		case 7: \
			launch_backprop_exact_kernel_const_const(7, block_size_const, single_output_feature_map_group); \
			break; \
		case 8: \
			launch_backprop_exact_kernel_const_const(8, block_size_const, single_output_feature_map_group); \
			break; \
		case 9: \
			launch_backprop_exact_kernel_const_const(9, block_size_const, single_output_feature_map_group); \
			break; \
		case 10: \
			launch_backprop_exact_kernel_const_const(10, block_size_const, single_output_feature_map_group); \
			break; \
		};

#define launch_backprop_exact_kernel(window_width, block_size, single_output_feature_map_group) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const(window_width, 1, single_output_feature_map_group); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const(window_width, 2, single_output_feature_map_group); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const(window_width, 3, single_output_feature_map_group); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const(window_width, 4, single_output_feature_map_group); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const(window_width, 5, single_output_feature_map_group); \
			break; \
		};

#define launch_backprop_kernel_const(block_size_const, single_output_feature_map_group) \
	convolution_3d_deriviative_tex_upd_kernel_kepler<block_size_const,single_output_feature_map_group><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, weights_tex, packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[0], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, backward_output_feature_map_group_size, entry_count, packed_config_count);

#define launch_backprop_kernel(block_size, single_output_feature_map_group) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_kernel_const(1, single_output_feature_map_group); \
			break; \
		case 2: \
			launch_backprop_kernel_const(2, single_output_feature_map_group); \
			break; \
		case 3: \
			launch_backprop_kernel_const(3, single_output_feature_map_group); \
			break; \
		case 4: \
			launch_backprop_kernel_const(4, single_output_feature_map_group); \
			break; \
		case 5: \
			launch_backprop_kernel_const(5, single_output_feature_map_group); \
			break; \
		};

#define launch_update_weights_exact_kernel_const(window_width_const, single_output_z_group_const) \
	convolution_3d_update_weights_exact_upd_kernel_kepler<window_width_const, single_output_z_group_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*data[0], input_tex, output_tex, *learning_rate[0], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, updater_output_z_group_count, texture_offset, entry_count, different_input, packed_config_count);

#define launch_update_weights_exact_kernel(window_width, single_output_z_group_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_update_weights_exact_kernel_const(1, single_output_z_group_const); \
			break; \
		case 2: \
			launch_update_weights_exact_kernel_const(2, single_output_z_group_const); \
			break; \
		case 3: \
			launch_update_weights_exact_kernel_const(3, single_output_z_group_const); \
			break; \
		case 4: \
			launch_update_weights_exact_kernel_const(4, single_output_z_group_const); \
			break; \
		case 5: \
			launch_update_weights_exact_kernel_const(5, single_output_z_group_const); \
			break; \
		case 6: \
			launch_update_weights_exact_kernel_const(6, single_output_z_group_const); \
			break; \
		case 7: \
			launch_update_weights_exact_kernel_const(7, single_output_z_group_const); \
			break; \
		case 8: \
			launch_update_weights_exact_kernel_const(8, single_output_z_group_const); \
			break; \
		case 9: \
			launch_update_weights_exact_kernel_const(9, single_output_z_group_const); \
			break; \
		case 10: \
			launch_update_weights_exact_kernel_const(10, single_output_z_group_const); \
			break; \
		};

#define launch_update_weights_kernel_const(single_output_z_group_const) \
	convolution_3d_update_weights_upd_kernel_kepler<single_output_z_group_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*data[0], input_tex, output_tex, *learning_rate[0], packed_config_list, output_configuration_specific.dimension_sizes[0], output_configuration_specific.dimension_sizes[1], output_configuration_specific.dimension_sizes[2], input_configuration_specific.dimension_sizes[0], input_configuration_specific.dimension_sizes[1], input_configuration_specific.dimension_sizes[2], window_sizes[0], window_sizes[1], window_sizes[2], input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, updater_output_z_group_count, texture_offset, entry_count, different_input, packed_config_count);

		void convolution_3d_layer_updater_cuda_kepler::enqueue_test(
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
			if (dynamic_memobjects[0] == 0)
				dynamic_memobjects[0] = cuda_texture_smart_ptr(new cuda_texture(input_neurons_buffer));
			cuda_texture& input_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[0].get()));
			int texture_offset = offset_input_entry_id * input_elem_count_per_entry;

			if (dynamic_memobjects[1] == 0)
				dynamic_memobjects[1] = cuda_texture_smart_ptr(new cuda_texture(data[0]));
			cuda_texture& weights_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[1].get()));

			if (forward_input_feature_map_group_count > 1)
				cuda_util::set_with_value(
					*cuda_config,
					*output_neurons_buffer,
					0.0F,
					output_elem_count_per_entry * entry_count,
					stream_id);

			int packed_config_count = forward_x_block_count * output_configuration_specific.dimension_sizes[1] * output_configuration_specific.dimension_sizes[2] * forward_output_feature_map_block_count * forward_input_feature_map_group_count;
			const packed_config<5> * packed_config_list = static_cast<const packed_config<5> *>((const void *)*additional_buffers[0]);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				packed_config_count,
				entry_count,
				1);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				if (forward_input_feature_map_group_count == 1)
				{
					launch_exact_kernel(window_sizes[0], forward_x_block_size, true);
				}
				else
				{
					launch_exact_kernel(window_sizes[0], forward_x_block_size, false);
				}
			}
			else
			{
				if (forward_input_feature_map_group_count == 1)
				{
					launch_kernel(forward_x_block_size, true);
				}
				else
				{
					launch_kernel(forward_x_block_size, false);
				}
			}
		}

		void convolution_3d_layer_updater_cuda_kepler::enqueue_backprop(
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
			if (!different_input)
				throw neural_network_exception("convolution_2d_layer_updater_cuda_kepler is not able to backprop to the same input");

			if (!backprop_required)
				throw neural_network_exception("convolution_2d_layer_updater_cuda_kepler is not configured to do backprop but requested to");

			if (dynamic_memobjects[2] == 0)
				dynamic_memobjects[2] = cuda_texture_smart_ptr(new cuda_texture(output_errors_buffer));
			cuda_texture& output_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[2].get()));

			if (dynamic_memobjects[1] == 0)
				dynamic_memobjects[1] = cuda_texture_smart_ptr(new cuda_texture(data[0]));
			cuda_texture& weights_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[1].get()));

			if (backward_output_feature_map_group_count > 1)
				cuda_util::set_with_value(
					*cuda_config,
					*input_errors_buffer,
					0.0F,
					input_elem_count_per_entry * entry_count,
					stream_id);

			int packed_config_count = backward_x_block_count * input_configuration_specific.dimension_sizes[1] * input_configuration_specific.dimension_sizes[2] * backward_input_feature_map_block_count * backward_output_feature_map_group_count;
			const packed_config<5> * packed_config_list = static_cast<const packed_config<5> *>((const void *)*additional_buffers[2]);

			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				packed_config_count,
				entry_count,
				1);

			if (window_sizes[0] <= MAX_WINDOW_WIDTH)
			{
				if (backward_output_feature_map_group_count == 1)
				{
					launch_backprop_exact_kernel(window_sizes[0], backward_x_block_size, true);
				}
				else
				{
					launch_backprop_exact_kernel(window_sizes[0], backward_x_block_size, false);
				}
			}
			else
			{
				if (backward_output_feature_map_group_count == 1)
				{
					launch_backprop_kernel(backward_x_block_size, true);
				}
				else
				{
					launch_backprop_kernel(backward_x_block_size, false);
				}
			}
		}

		void convolution_3d_layer_updater_cuda_kepler::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& learning_rate,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			// Update biases
			{
				int threadblock_size = get_threadblock_size_biases(output_elem_count_per_feature_map);
				dim3 grid_size(1, output_configuration_specific.feature_map_count, entry_count);
				dim3 block_size(threadblock_size, 1, 1);
				int min_iteration_count = output_elem_count_per_feature_map / threadblock_size;

				convolution_3d_update_biases_upd_kernel_kepler<<<grid_size, block_size, 0, stream_id>>>(
					*data[1],
					*output_errors_buffer,
					*learning_rate[1],
					output_configuration_specific.feature_map_count,
					output_elem_count_per_feature_map,
					min_iteration_count);
			}

			if (dynamic_memobjects[2] == 0)
				dynamic_memobjects[2] = cuda_texture_smart_ptr(new cuda_texture(output_errors_buffer));
			cuda_texture& output_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[2].get()));

			if (dynamic_memobjects[0] == 0)
				dynamic_memobjects[0] = cuda_texture_smart_ptr(new cuda_texture(input_neurons_buffer));
			cuda_texture& input_tex = *(dynamic_cast<cuda_texture *>(dynamic_memobjects[0].get()));
			int texture_offset = offset_input_entry_id * input_elem_count_per_entry;

			int packed_config_count = window_sizes[1] * window_sizes[2] * updater_output_z_group_count * updater_output_feature_map_block_count * input_configuration_specific.feature_map_count;
			const packed_config<5> * packed_config_list = static_cast<const packed_config<5> *>((const void *)*additional_buffers[1]);

			// Update weights
			{
				if (updater_window_x_block_count == 1)
				{
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						packed_config_count,
						entry_count,
						1);

					if (updater_output_z_group_count == 1)
					{
						launch_update_weights_exact_kernel(window_sizes[0], true);
					}
					else
					{
						launch_update_weights_exact_kernel(window_sizes[0], false);
					}
				}
				else
				{
					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						updater_window_x_block_count,
						packed_config_count,
						entry_count);

					if (updater_output_z_group_count == 1)
					{
						launch_update_weights_kernel_const(true);
					}
					else
					{
						launch_update_weights_kernel_const(false);
					}
				}
			}
		}

		int convolution_3d_layer_updater_cuda_kepler::get_block_size(int width)
		{
			int block_count = (width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
			int block_size = (width + block_count - 1) / block_count;
			return block_size;
		}

		void convolution_3d_layer_updater_cuda_kepler::updater_configured()
		{
			std::tr1::shared_ptr<const convolution_layer> layer_derived = std::tr1::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				window_sizes.push_back(static_cast<int>(*it));

			forward_x_block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
			forward_x_block_count = (output_configuration_specific.dimension_sizes[0] + forward_x_block_size - 1) / forward_x_block_size;
			forward_output_feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

			updater_output_feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;
			updater_window_x_block_count = (window_sizes[0] <= MAX_WINDOW_WIDTH) ? 1 : (window_sizes[0] + WINDOW_WIDTH_LOCAL - 1) / WINDOW_WIDTH_LOCAL;
			{
				std::tr1::array<int, 3> size_list;
				size_list[0] = window_sizes[1];
				size_list[1] = window_sizes[2];
				size_list[2] = input_configuration_specific.feature_map_count;
				space_filling_curve<3>::fill_pattern(size_list, updater_config_ordered_list1);
			}

			if (backprop_required)
			{
				backward_x_block_size = get_block_size(input_configuration_specific.dimension_sizes[0]);
				backward_x_block_count = (input_configuration_specific.dimension_sizes[0] + backward_x_block_size - 1) / backward_x_block_size;
				backward_input_feature_map_block_count = (input_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;
			}
		}

		bool convolution_3d_layer_updater_cuda_kepler::is_in_place_backprop() const
		{
			return false;
		}

		std::vector<unsigned int> convolution_3d_layer_updater_cuda_kepler::get_linear_addressing_through_texture_per_entry() const
		{
			std::vector<unsigned int> res;

			res.push_back(input_elem_count_per_entry);
			res.push_back(output_elem_count_per_entry);

			return res;
		}

		int convolution_3d_layer_updater_cuda_kepler::get_threadblock_size_biases(int output_neuron_count)
		{
			int threadblock_size;

			if (output_neuron_count < 128)
			{
				threadblock_size = (output_neuron_count + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (output_neuron_count + 128 - 1) / 128;
				threadblock_size = (output_neuron_count + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}

		std::vector<size_t> convolution_3d_layer_updater_cuda_kepler::get_sizes_of_additional_buffers_fixed() const
		{
			std::vector<size_t> res;

			res.push_back(sizeof(packed_config<5>) * forward_x_block_count * output_configuration_specific.dimension_sizes[1] * output_configuration_specific.dimension_sizes[2] * input_configuration_specific.feature_map_count * forward_output_feature_map_block_count);

			res.push_back(sizeof(packed_config<5>) * window_sizes[1] * window_sizes[2] * output_configuration_specific.dimension_sizes[1] * input_configuration_specific.feature_map_count * updater_output_feature_map_block_count);

			if (backprop_required)
				res.push_back(sizeof(packed_config<5>) * backward_x_block_count * input_configuration_specific.dimension_sizes[1] * input_configuration_specific.dimension_sizes[2] * output_configuration_specific.feature_map_count * backward_input_feature_map_block_count);

			return res;
		}

		void convolution_3d_layer_updater_cuda_kepler::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
			{
				std::vector<packed_config<5> > task_list;
				packed_config<5> new_elem;

				for(int input_feature_map_group_id = 0; input_feature_map_group_id < forward_input_feature_map_group_count; ++input_feature_map_group_id)
				{
					new_elem.set_val(4, input_feature_map_group_id * forward_input_feature_map_group_size);
					for(int output_feature_map_block_id = 0; output_feature_map_block_id < forward_output_feature_map_block_count; ++output_feature_map_block_id)
					{
						new_elem.set_val(3, output_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
						for(int z = 0; z < output_configuration_specific.dimension_sizes[2]; ++z)
						{
							new_elem.set_val(2, z);
							for(int y = 0; y < output_configuration_specific.dimension_sizes[1]; ++y)
							{
								new_elem.set_val(1, y);
								for(int x = 0; x < forward_x_block_count; ++x)
								{
									new_elem.set_val(0, x * forward_x_block_size);
									task_list.push_back(new_elem);
								}
							}
						}
					}
				}

				cuda_safe_call(cudaMemcpy(*additional_buffers[0], &(*task_list.begin()), sizeof(packed_config<5>) * task_list.size(), cudaMemcpyHostToDevice));
			}

			{
				std::vector<packed_config<5> > task_list;
				packed_config<5> new_elem;

				for(std::vector<std::tr1::array<int, 2> >::const_iterator it2 = updater_config_ordered_list2.begin(); it2 != updater_config_ordered_list2.end(); ++it2)
				{
					new_elem.set_val(3, it2->at(0) * FEATURE_MAP_BLOCK_SIZE); 
					new_elem.set_val(4, it2->at(1));
					for(std::vector<std::tr1::array<int, 3> >::const_iterator it1 = updater_config_ordered_list1.begin(); it1 != updater_config_ordered_list1.end(); ++it1)
					{
						new_elem.set_val(0, it1->at(0));
						new_elem.set_val(1, it1->at(1));
						new_elem.set_val(2, it1->at(2));
						task_list.push_back(new_elem);
					}
				}

				cuda_safe_call(cudaMemcpy(*additional_buffers[1], &(*task_list.begin()), sizeof(packed_config<5>) * task_list.size(), cudaMemcpyHostToDevice));
			}

			if (backprop_required)
			{
				std::vector<packed_config<5> > task_list;
				packed_config<5> new_elem;

				for(int output_feature_map_group_id = 0; output_feature_map_group_id < backward_output_feature_map_group_count; ++output_feature_map_group_id)
				{
					new_elem.set_val(4, output_feature_map_group_id * backward_output_feature_map_group_size);
					for(int input_feature_map_block_id = 0; input_feature_map_block_id < backward_input_feature_map_block_count; ++input_feature_map_block_id)
					{
						new_elem.set_val(3, input_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
						for(int z = 0; z < input_configuration_specific.dimension_sizes[2]; ++z)
						{
							new_elem.set_val(2, z);
							for(int y = 0; y < input_configuration_specific.dimension_sizes[1]; ++y)
							{
								new_elem.set_val(1, y);
								for(int x = 0; x < backward_x_block_count; ++x)
								{
									new_elem.set_val(0, x * backward_x_block_size + (backward_x_block_size - 1));
									task_list.push_back(new_elem);
								}
							}
						}
					}
				}

				cuda_safe_call(cudaMemcpy(*additional_buffers[2], &(*task_list.begin()), sizeof(packed_config<5>) * task_list.size(), cudaMemcpyHostToDevice));
			}
		}

		void convolution_3d_layer_updater_cuda_kepler::set_max_entry_count(unsigned int max_entry_count)
		{
			forward_input_feature_map_group_count = cuda_util::get_group_count(
				*cuda_config,
				forward_x_block_count * output_configuration_specific.dimension_sizes[1] * output_configuration_specific.dimension_sizes[2] * forward_output_feature_map_block_count * max_entry_count,
				input_configuration_specific.feature_map_count);
			forward_input_feature_map_group_size = (input_configuration_specific.feature_map_count + forward_input_feature_map_group_count - 1) / forward_input_feature_map_group_count;

			updater_output_z_group_count = cuda_util::get_group_count(
				*cuda_config,
				updater_output_feature_map_block_count * input_configuration_specific.feature_map_count * max_entry_count * updater_window_x_block_count * window_sizes[1] * window_sizes[2],
				output_configuration_specific.dimension_sizes[2]);
			updater_output_z_group_size = (output_configuration_specific.dimension_sizes[2] + updater_output_z_group_count - 1) / updater_output_z_group_count;
			{
				std::tr1::array<int, 2> size_list;
				size_list[0] = updater_output_feature_map_block_count;
				size_list[1] = updater_output_z_group_count;
				space_filling_curve<2>::fill_pattern(size_list, updater_config_ordered_list2);
			}

			if (backprop_required)
			{
				backward_output_feature_map_group_count = cuda_util::get_group_count(
					*cuda_config,
					backward_x_block_count * input_configuration_specific.dimension_sizes[1] * input_configuration_specific.dimension_sizes[2] * backward_input_feature_map_block_count * max_entry_count,
					output_configuration_specific.feature_map_count);
				backward_output_feature_map_group_size = (output_configuration_specific.feature_map_count + backward_output_feature_map_group_count - 1) / backward_output_feature_map_group_count;
			}
		}

		int convolution_3d_layer_updater_cuda_kepler::get_dynamic_memobject_count() const
		{
			return 3;
		}
	}
}
