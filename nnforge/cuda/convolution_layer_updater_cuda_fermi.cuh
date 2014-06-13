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
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "space_filling_curve.h"
#include "sequential_curve.h"

#include "../convolution_layer.h"
#include "../nn_types.h"

#define FEATURE_MAP_BLOCK_SIZE 4
#define WINDOW_WIDTH_LOCAL 4
#define MAX_BLOCK_SIZE 5
#define MAX_WINDOW_WIDTH 10

namespace nnforge
{
	namespace cuda
	{
		texture<float2, cudaTextureType1D, cudaReadModeElementType> input_tex_ref;
		texture<float2, cudaTextureType1D, cudaReadModeElementType> output_tex_ref;

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int BLOCK_SIZE, bool single_input_feature_map_group>
		__launch_bounds__(256, 2)
		__global__ void convolution_tex_exact_blocked_upd_kernel_fermi(
			float * __restrict output,
			const float2 * __restrict weights,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count,
			int input_feature_map_group_size,
			bool different_input,
			int weight_elem_count_striped_per_entry)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int xyzw[DIMENSION_COUNT];
				int total_weight_count = window_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					total_weight_count *= window_sizes[i];
				int weight_count_per_output_feature_map = input_feature_map_count_striped * total_weight_count;
				packed_config<DIMENSION_COUNT+2> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i);
				int output_feature_map_id = conf.get_val(DIMENSION_COUNT);
				int base_input_feature_map_id = conf.get_val(DIMENSION_COUNT + 1);
				int input_elem_id = (different_input ? entry_id * input_feature_map_count_striped : 0) + base_input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i];
				const float2 * current_weights = weights + (int)((output_feature_map_id * input_feature_map_count_striped + base_input_feature_map_id) * total_weight_count + entry_id * weight_elem_count_striped_per_entry);
				int iteration_count = min(input_feature_map_group_size, input_feature_map_count_striped - base_input_feature_map_id);

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
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < iteration_count; ++input_layer_id)
				{
					for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? window_sizes[3] : 1); ++input_w)
					{
						for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? window_sizes[2] : 1); ++input_z)
						{
							for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? window_sizes[1] : 1); ++input_y)
							{
								#pragma unroll
								for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
								{
									float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										weight_list[i] = current_weights[weight_offsets[i]];
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										float2 inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									current_weights++;
									input_elem_id++;
								} // input_x
								input_elem_id += input_sizes[0] - WINDOW_WIDTH;
							} // for input_y
							if (DIMENSION_COUNT > 1)
								input_elem_id += input_sizes[0] * (input_sizes[1] - window_sizes[1]);
						} // for input_z
						if (DIMENSION_COUNT > 2)
							input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - window_sizes[2]);
					} // for input_w
					if (DIMENSION_COUNT > 3)
						input_elem_id += input_sizes[2] * input_sizes[1] * input_sizes[0] * (input_sizes[3] - window_sizes[3]);
				}

				int output_offset = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_offset = output_offset * output_sizes[i] + xyzw[i];
				float * base_output = output + output_offset;
				int output_neuron_count_per_feature_map = output_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					output_neuron_count_per_feature_map *= output_sizes[i];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_sizes[0] - xyzw[0])
							{
								if (single_input_feature_map_group)
								{
									base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
								}
								else
								{
									atomicAdd(base_output + output_neuron_count_per_feature_map * i + j, sums[i * BLOCK_SIZE + j]);
								}
							}
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int BLOCK_SIZE, bool single_input_feature_map_group>
		__launch_bounds__(256, 2)
		__global__ void convolution_tex_generic_blocked_upd_kernel_fermi(
			float * __restrict output,
			const float2 * __restrict weights,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count,
			int input_feature_map_group_size,
			bool different_input,
			int weight_elem_count_striped_per_entry)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int xyzw[DIMENSION_COUNT];
				int total_weight_count = window_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					total_weight_count *= window_sizes[i];
				int weight_count_per_output_feature_map = input_feature_map_count_striped * total_weight_count;
				packed_config<DIMENSION_COUNT+2> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i);
				int output_feature_map_id = conf.get_val(DIMENSION_COUNT);
				int base_input_feature_map_id = conf.get_val(DIMENSION_COUNT + 1);
				int input_elem_id = (different_input ? entry_id * input_feature_map_count_striped : 0) + base_input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i];
				const float2 * current_weights = weights + (int)((output_feature_map_id * input_feature_map_count_striped + base_input_feature_map_id) * total_weight_count + entry_id * weight_elem_count_striped_per_entry);
				int iteration_count = min(input_feature_map_group_size, input_feature_map_count_striped - base_input_feature_map_id);

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
				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
					weight_offsets[i] = (i < output_feature_map_count - output_feature_map_id) ? weight_count_per_output_feature_map * i : 0;

				for(int input_layer_id = 0; input_layer_id < iteration_count; ++input_layer_id)
				{
					for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? window_sizes[3] : 1); ++input_w)
					{
						for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? window_sizes[2] : 1); ++input_z)
						{
							for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? window_sizes[1] : 1); ++input_y)
							{
								#pragma unroll 4
								for(int input_x = 0; input_x < window_sizes[0]; ++input_x)
								{
									float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										weight_list[i] = current_weights[weight_offsets[i]];
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										float2 inp = tex1Dfetch(input_tex_ref, input_elem_id + j); 
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									current_weights++;
									input_elem_id++;
								} // for input_x
								input_elem_id += input_sizes[0] - window_sizes[0];
							} // for input_y
							if (DIMENSION_COUNT > 1)
								input_elem_id += input_sizes[0] * (input_sizes[1] - window_sizes[1]);
						} // for input_z
						if (DIMENSION_COUNT > 2)
							input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - window_sizes[2]);
					} // for input_w
					if (DIMENSION_COUNT > 3)
						input_elem_id += input_sizes[2] * input_sizes[1] * input_sizes[0] * (input_sizes[3] - window_sizes[3]);
				}

				int output_offset = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_offset = output_offset * output_sizes[i] + xyzw[i];
				float * base_output = output + output_offset;
				int output_neuron_count_per_feature_map = output_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					output_neuron_count_per_feature_map *= output_sizes[i];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < output_feature_map_count - output_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j < output_sizes[0] - xyzw[0])
							{
								if (single_input_feature_map_group)
								{
									base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
								}
								else
								{
									atomicAdd(base_output + output_neuron_count_per_feature_map * i + j, sums[i * BLOCK_SIZE + j]);
								}
							}
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int BLOCK_SIZE, bool single_output_feature_map_group>
		__launch_bounds__(256, 2)
		__global__ void convolution_backprop_tex_exact_blocked_upd_kernel_fermi(
			float * __restrict input_errors,
			const float2 * __restrict weights,
			const packed_config<DIMENSION_COUNT+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int output_feature_map_count_striped,
			int entry_count,
			int packed_config_count,
			int output_feature_map_group_size,
			int weight_elem_count_striped_per_entry)
		{
			int xyzw[DIMENSION_COUNT];
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int total_weight_count = window_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					total_weight_count *= window_sizes[i];
				int weight_count_per_striped_input_feature_map = total_weight_count;
				int weight_count_per_output_feature_map = input_feature_map_count_striped * total_weight_count;
				packed_config<DIMENSION_COUNT+2> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i);
				int input_feature_map_id_striped = conf.get_val(DIMENSION_COUNT);
				int base_output_feature_map_id_striped = conf.get_val(DIMENSION_COUNT + 1);
				int output_elem_id = entry_id * output_feature_map_count_striped + base_output_feature_map_id_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw[i];
				const float2 * current_weights = weights + (int)(((base_output_feature_map_id_striped << 1) * input_feature_map_count_striped + input_feature_map_id_striped) * total_weight_count + entry_id * weight_elem_count_striped_per_entry);
				int iteration_count = min(output_feature_map_group_size, output_feature_map_count_striped - base_output_feature_map_id_striped);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
				{
					weight_offsets[i] = (i < input_feature_map_count_striped - input_feature_map_id_striped) ? weight_count_per_striped_input_feature_map * i : 0;
					weight_offsets[i + (FEATURE_MAP_BLOCK_SIZE/2)] = weight_offsets[i] + weight_count_per_output_feature_map;
				}

				int min_exclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					min_exclusive[i] = xyzw[i] - output_sizes[i];
				int max_inclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					max_inclusive[i] = xyzw[i];

				for(int output_layer_id = 0; output_layer_id < iteration_count; ++output_layer_id)
				{
					for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((input_w > min_exclusive[3]) && (input_w <= max_inclusive[3])) : true;
						for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && (input_z > min_exclusive[2]) && (input_z <= max_inclusive[2])) : true;
							for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? window_sizes[1] : 1); ++input_y)
							{
								bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && (input_y > min_exclusive[1]) && (input_y <= max_inclusive[1])) : true;

								float2 output_vals[BLOCK_SIZE + WINDOW_WIDTH - 1];
								#pragma unroll
								for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH - 1; ++i)
								{
									bool b_fit0 = b_fit1 && (i > min_exclusive[0]) && (i <= max_inclusive[0]);
									output_vals[i] = tex1Dfetch(output_tex_ref, b_fit0 ? (output_elem_id - i) : -1);
								}

								#pragma unroll
								for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
								{
									float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
									{
										weight_list[i] = current_weights[weight_offsets[i]];
										weight_list[i + (FEATURE_MAP_BLOCK_SIZE/2)] = current_weights[weight_offsets[i + (FEATURE_MAP_BLOCK_SIZE/2)]];
									}

									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
										{
											sums[(i * 2) * BLOCK_SIZE + j] += output_vals[input_x + j].x * weight_list[i].x;
											sums[(i * 2) * BLOCK_SIZE + j] += output_vals[input_x + j].y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].x;
											sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_vals[input_x + j].x * weight_list[i].y;
											sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_vals[input_x + j].y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].y;
										}
									}
									current_weights++;
								}
								if (DIMENSION_COUNT == 1)
									output_elem_id += output_sizes[0];
								else
									output_elem_id -= output_sizes[0];
							} // for(int input_y
							if (DIMENSION_COUNT == 2)
								output_elem_id += output_sizes[0] * (window_sizes[1] + output_sizes[1]);
							else if (DIMENSION_COUNT > 2)
								output_elem_id += output_sizes[0] * (window_sizes[1] - output_sizes[1]);
						} // for(int input_z
						if (DIMENSION_COUNT == 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (window_sizes[2] + output_sizes[2]);
						else if (DIMENSION_COUNT > 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (window_sizes[2] - output_sizes[2]);
					} // for(int input_w
					if (DIMENSION_COUNT == 4)
						output_elem_id += output_sizes[2] * output_sizes[1] * output_sizes[0] * (window_sizes[3] + output_sizes[3]);
					current_weights += (weight_count_per_output_feature_map << 1) - weight_count_per_striped_input_feature_map;
				} // for(int output_layer_id

				int input_feature_map_id = input_feature_map_id_striped << 1;
				int input_offset = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_offset = input_offset * input_sizes[i] + xyzw[i];
				float * base_input = input_errors + input_offset;
				int input_neuron_count_per_feature_map = input_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					input_neuron_count_per_feature_map *= input_sizes[i];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < input_feature_map_count - input_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j > xyzw[0] - input_sizes[0])
							{
								if (single_output_feature_map_group)
								{
									*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
								}
								else
								{
									atomicAdd(base_input + input_neuron_count_per_feature_map * i - j, sums[i * BLOCK_SIZE + j]);
								}
							}
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int BLOCK_SIZE, bool single_output_feature_map_group>
		__launch_bounds__(256, 2)
		__global__ void convolution_backprop_tex_generic_blocked_upd_kernel_fermi(
			float * __restrict input_errors,
			const float2 * __restrict weights,
			const packed_config<DIMENSION_COUNT+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int output_feature_map_count_striped,
			int entry_count,
			int packed_config_count,
			int output_feature_map_group_size,
			int weight_elem_count_striped_per_entry)
		{
			int xyzw[DIMENSION_COUNT];
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int total_weight_count = window_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					total_weight_count *= window_sizes[i];
				int weight_count_per_striped_input_feature_map = total_weight_count;
				int weight_count_per_output_feature_map = input_feature_map_count_striped * total_weight_count;
				packed_config<DIMENSION_COUNT+2> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i);
				int input_feature_map_id_striped = conf.get_val(DIMENSION_COUNT);
				int base_output_feature_map_id_striped = conf.get_val(DIMENSION_COUNT + 1);
				int output_elem_id = entry_id * output_feature_map_count_striped + base_output_feature_map_id_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw[i];
				const float2 * current_weights = weights + (int)(((base_output_feature_map_id_striped << 1) * input_feature_map_count_striped + input_feature_map_id_striped) * total_weight_count + entry_id * weight_elem_count_striped_per_entry);
				int iteration_count = min(output_feature_map_group_size, output_feature_map_count_striped - base_output_feature_map_id_striped);

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int weight_offsets[FEATURE_MAP_BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
				{
					weight_offsets[i] = (i < input_feature_map_count_striped - input_feature_map_id_striped) ? weight_count_per_striped_input_feature_map * i : 0;
					weight_offsets[i + (FEATURE_MAP_BLOCK_SIZE/2)] = weight_offsets[i] + weight_count_per_output_feature_map;
				}

				int min_exclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					min_exclusive[i] = xyzw[i] - output_sizes[i];
				int max_inclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					max_inclusive[i] = xyzw[i];

				for(int output_layer_id = 0; output_layer_id < iteration_count; ++output_layer_id)
				{
					for(int input_w = 0; input_w < (DIMENSION_COUNT > 3 ? window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((input_w > min_exclusive[3]) && (input_w <= max_inclusive[3])) : true;
						for(int input_z = 0; input_z < (DIMENSION_COUNT > 2 ? window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && (input_z > min_exclusive[2]) && (input_z <= max_inclusive[2])) : true;
							for(int input_y = 0; input_y < (DIMENSION_COUNT > 1 ? window_sizes[1] : 1); ++input_y)
							{
								bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && (input_y > min_exclusive[1]) && (input_y <= max_inclusive[1])) : true;

								int input_x = 0;
								#pragma unroll 1
								for(; input_x < (window_sizes[0] - (WINDOW_WIDTH_LOCAL - 1)); input_x += WINDOW_WIDTH_LOCAL)
								{
									float2 output_vals[BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1];
									#pragma unroll
									for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH_LOCAL - 1; ++i)
									{
										bool b_fit0 = b_fit1 && (i > min_exclusive[0]) && (i <= max_inclusive[0]);
										output_vals[i] = tex1Dfetch(output_tex_ref, b_fit0 ? (output_elem_id - i) : -1);
									}
									output_elem_id -= WINDOW_WIDTH_LOCAL;

									#pragma unroll
									for(int input_x_local = 0; input_x_local < WINDOW_WIDTH_LOCAL; ++input_x_local)
									{
										float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
										{
											weight_list[i] = current_weights[weight_offsets[i]];
											weight_list[i + (FEATURE_MAP_BLOCK_SIZE/2)] = current_weights[weight_offsets[i + (FEATURE_MAP_BLOCK_SIZE/2)]];
										}

										#pragma unroll
										for(int j = 0; j < BLOCK_SIZE; ++j)
										{
											#pragma unroll
											for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
											{
												sums[(i * 2) * BLOCK_SIZE + j] += output_vals[input_x_local + j].x * weight_list[i].x;
												sums[(i * 2) * BLOCK_SIZE + j] += output_vals[input_x_local + j].y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].x;
												sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_vals[input_x_local + j].x * weight_list[i].y;
												sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_vals[input_x_local + j].y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].y;
											}
										}
										current_weights++;
									}
								}
								#pragma unroll 1
								for(; input_x < window_sizes[0]; ++input_x)
								{
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										bool b_fit0 = b_fit1 && (input_x + j > min_exclusive[0]) && (input_x + j <= max_inclusive[0]);
										float2 output_val = tex1Dfetch(output_tex_ref, b_fit0 ? (output_elem_id - j) : -1);
										float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
										{
											weight_list[i] = current_weights[weight_offsets[i]];
											weight_list[i + (FEATURE_MAP_BLOCK_SIZE/2)] = current_weights[weight_offsets[i + (FEATURE_MAP_BLOCK_SIZE/2)]];
										}
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE/2; ++i)
										{
											sums[(i * 2) * BLOCK_SIZE + j] += output_val.x * weight_list[i].x;
											sums[(i * 2) * BLOCK_SIZE + j] += output_val.y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].x;
											sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_val.x * weight_list[i].y;
											sums[(i * 2 + 1) * BLOCK_SIZE + j] += output_val.y * weight_list[(FEATURE_MAP_BLOCK_SIZE/2) + i].y;
										}
									}
									current_weights++;
									output_elem_id--;
								}
								if (DIMENSION_COUNT == 1)
									output_elem_id += window_sizes[0] + output_sizes[0];
								else
									output_elem_id += window_sizes[0] - output_sizes[0];
							} // for(int input_y
							if (DIMENSION_COUNT == 2)
								output_elem_id += output_sizes[0] * (window_sizes[1] + output_sizes[1]);
							else if (DIMENSION_COUNT > 2)
								output_elem_id += output_sizes[0] * (window_sizes[1] - output_sizes[1]);
						} // for(int input_z
						if (DIMENSION_COUNT == 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (window_sizes[2] + output_sizes[2]);
						else if (DIMENSION_COUNT > 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (window_sizes[2] - output_sizes[2]);
					} // for(int input_w
					if (DIMENSION_COUNT == 4)
						output_elem_id += output_sizes[2] * output_sizes[1] * output_sizes[0] * (window_sizes[3] + output_sizes[3]);
					current_weights += (weight_count_per_output_feature_map << 1) - weight_count_per_striped_input_feature_map;
				} // for(int output_layer_id

				int input_feature_map_id = input_feature_map_id_striped << 1;
				int input_offset = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_offset = input_offset * input_sizes[i] + xyzw[i];
				float * base_input = input_errors + input_offset;
				int input_neuron_count_per_feature_map = input_sizes[0];
				#pragma unroll
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					input_neuron_count_per_feature_map *= input_sizes[i];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
				{
					if (i < input_feature_map_count - input_feature_map_id)
					{
						#pragma unroll
						for(int j = 0; j < BLOCK_SIZE; ++j)
						{
							if (j > xyzw[0] - input_sizes[0])
							{
								if (single_output_feature_map_group)
								{
									*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
								}
								else
								{
									atomicAdd(base_input + input_neuron_count_per_feature_map * i - j, sums[i * BLOCK_SIZE + j]);
								}
							}
						}
					}
				}
			}
		}

		extern __shared__ float arr_sh[];
		__global__ void convolution_update_biases_upd_kernel_fermi(
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

			volatile float * arr = arr_sh;
			arr[thread_id] = sum;
			int lane_id = thread_id & 31;
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			}
			sum = arr[thread_id];

			if (lane_id == 0)
			{
				int offset = entry_id * output_feature_map_count + output_feature_map_id;
				float current_learning_rate_val = learning_rate[offset];
				atomicAdd(biases + offset, sum * current_learning_rate_val);
			}
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, bool single_elem_per_destination>
		__launch_bounds__(256, 2)
		__global__ void convolution_update_weights_exact_upd_kernel_fermi(
			float2 * __restrict weights,
			const float2 * __restrict output_errors,
			const float2 * __restrict learning_rate,
			const packed_config<DIMENSION_COUNT*2+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_feature_map_count_striped,
			int output_feature_map_count_striped,
			int input_elem_count_per_entry_striped,
			int output_elem_count_per_entry_striped,
			int entry_count,
			bool different_input,
			int packed_config_count,
			int last_dimension_group_count,
			int weight_elem_count_striped_per_entry,
			float weight_decay)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (packed_config_id < packed_config_count) && (entry_id < entry_count);
			if (in_bounds)
			{
				packed_config<DIMENSION_COUNT*2+2> conf = packed_config_list[packed_config_id];
				int weight_xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_xyzw[i] = conf.get_val(i);
				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i + DIMENSION_COUNT);
				int input_feature_map_striped_id = conf.get_val(DIMENSION_COUNT * 2);
				int output_feature_map_striped_id = conf.get_val(DIMENSION_COUNT * 2 + 1);

				int output_errors_offset = entry_id * output_feature_map_count_striped + output_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_errors_offset = output_errors_offset * output_sizes[i] + xyzw[i];
				const float2 * current_output_errors = output_errors + output_errors_offset;

				int input_elem_id = (different_input ? entry_id * input_feature_map_count_striped : 0) + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i] + weight_xyzw[i];

				float sums[WINDOW_WIDTH * 4];
				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH * 4; ++i)
					sums[i] = 0.0F;

				int iteration_count_x = output_sizes[0];

				int output_shift = last_dimension_group_count * output_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT - 1; ++i)
					output_shift *= output_sizes[i];
				output_shift -= iteration_count_x;

				int input_shift = last_dimension_group_count * input_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT - 1; ++i)
					input_shift *= input_sizes[i];
				input_shift -= (iteration_count_x + (WINDOW_WIDTH - 1));

				for(int t = (DIMENSION_COUNT > 1 ? xyzw[DIMENSION_COUNT - 1] : 0); t < (DIMENSION_COUNT > 1 ? output_sizes[DIMENSION_COUNT - 1] : 1); t += (DIMENSION_COUNT > 1 ? last_dimension_group_count : 1))
				{
					float2 input_buf[WINDOW_WIDTH];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH; ++i)
					{
						input_buf[i] = tex1Dfetch(input_tex_ref, input_elem_id);
						++input_elem_id;
					}

					for(int x = 0; x < iteration_count_x; ++x)
					{
						float2 output_error = *current_output_errors;

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH - 1; ++i)
							input_buf[i] = input_buf[i + 1];
						input_buf[WINDOW_WIDTH - 1] = tex1Dfetch(input_tex_ref, input_elem_id);

						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH; ++j)
						{
							sums[j * 4] += output_error.x * input_buf[j].x;
							sums[j * 4 + 1] += output_error.x * input_buf[j].y;
							sums[j * 4 + 2] += output_error.y * input_buf[j].x;
							sums[j * 4 + 3] += output_error.y * input_buf[j].y;
						}

						current_output_errors++;
						input_elem_id++;
					}

					current_output_errors += output_shift;
					input_elem_id += input_shift;
				}

				int output_feature_map_id = output_feature_map_striped_id * 2;
				int weights_offset = output_feature_map_id * input_feature_map_count_striped + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					weights_offset = weights_offset * window_sizes[i] + weight_xyzw[i];
				weights_offset += weight_elem_count_striped_per_entry * entry_id;
				int weight_count_per_feature_map_pair = window_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					weight_count_per_feature_map_pair *= window_sizes[i];

				#pragma unroll
				for(int i = 0; i < 2; ++i)
				{
					if (output_feature_map_id + i < output_feature_map_count)
					{
						int weights_offset1 = weights_offset + i * (input_feature_map_count_striped * weight_count_per_feature_map_pair);
						#pragma unroll
						for(int k = 0; k < WINDOW_WIDTH; ++k)
						{
							int weights_offset3 = weights_offset1 + k;
							float2 lr = learning_rate[weights_offset3];
							if (single_elem_per_destination)
							{
								float2 current_w = weights[weights_offset3];
								float new_val1 = current_w.x + lr.x * (sums[k * 4 + i * 2] - weight_decay * current_w.x);
								float new_val2 = current_w.y + lr.y * (sums[k * 4 + i * 2 + 1] - weight_decay * current_w.y);
								weights[weights_offset3] = make_float2(new_val1, new_val2);
							}
							else
							{
								float upd_val1 = lr.x * sums[k * 4 + i * 2];
								float upd_val2 = lr.y * sums[k * 4 + i * 2 + 1];
								atomicAdd((float *)(weights + weights_offset3), upd_val1);
								atomicAdd((float *)(weights + weights_offset3) + 1, upd_val2);
							}
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, bool single_elem_per_destination>
		__launch_bounds__(256, 2)
		__global__ void convolution_update_weights_generic_upd_kernel_fermi(
			float2 * __restrict weights,
			const float2 * __restrict output_errors,
			const float2 * __restrict learning_rate,
			const packed_config<DIMENSION_COUNT*2+2> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_feature_map_count_striped,
			int output_feature_map_count_striped,
			int input_elem_count_per_entry_striped,
			int output_elem_count_per_entry_striped,
			int entry_count,
			bool different_input,
			int packed_config_count,
			int last_dimension_group_count,
			int weight_elem_count_striped_per_entry,
			float weight_decay)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (packed_config_id < packed_config_count) && (entry_id < entry_count);
			if (in_bounds)
			{
				packed_config<DIMENSION_COUNT*2+2> conf = packed_config_list[packed_config_id];
				int weight_xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_xyzw[i] = conf.get_val(i);
				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					xyzw[i] = conf.get_val(i + DIMENSION_COUNT);
				int input_feature_map_striped_id = conf.get_val(DIMENSION_COUNT * 2);
				int output_feature_map_striped_id = conf.get_val(DIMENSION_COUNT * 2 + 1);

				int output_errors_offset = entry_id * output_feature_map_count_striped + output_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_errors_offset = output_errors_offset * output_sizes[i] + xyzw[i];
				const float2 * current_output_errors = output_errors + output_errors_offset;

				int input_elem_id = (different_input ? entry_id * input_feature_map_count_striped : 0) + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i] + weight_xyzw[i];

				float sums[WINDOW_WIDTH_LOCAL * 4];
				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH_LOCAL * 4; ++i)
					sums[i] = 0.0F;

				int iteration_count_x = output_sizes[0];

				int output_shift = last_dimension_group_count * output_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT - 1; ++i)
					output_shift *= output_sizes[i];
				output_shift -= iteration_count_x;

				int input_shift = last_dimension_group_count * input_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT - 1; ++i)
					input_shift *= input_sizes[i];
				input_shift -= (iteration_count_x + (WINDOW_WIDTH_LOCAL - 1));

				for(int t = (DIMENSION_COUNT > 1 ? xyzw[DIMENSION_COUNT - 1] : 0); t < (DIMENSION_COUNT > 1 ? output_sizes[DIMENSION_COUNT - 1] : 1); t += (DIMENSION_COUNT > 1 ? last_dimension_group_count : 1))
				{
					float2 input_buf[WINDOW_WIDTH_LOCAL];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH_LOCAL; ++i)
					{
						input_buf[i] = tex1Dfetch(input_tex_ref, input_elem_id);
						++input_elem_id;
					}

					for(int x = 0; x < iteration_count_x; ++x)
					{
						float2 output_error = *current_output_errors;

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH_LOCAL - 1; ++i)
							input_buf[i] = input_buf[i + 1];
						input_buf[WINDOW_WIDTH_LOCAL - 1] = tex1Dfetch(input_tex_ref, input_elem_id);

						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
						{
							sums[j * 4] += output_error.x * input_buf[j].x;
							sums[j * 4 + 1] += output_error.x * input_buf[j].y;
							sums[j * 4 + 2] += output_error.y * input_buf[j].x;
							sums[j * 4 + 3] += output_error.y * input_buf[j].y;
						}

						current_output_errors++;
						input_elem_id++;
					}

					current_output_errors += output_shift;
					input_elem_id += input_shift;
				}

				int output_feature_map_id = output_feature_map_striped_id * 2;
				int weights_offset = output_feature_map_id * input_feature_map_count_striped + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					weights_offset = weights_offset * window_sizes[i] + weight_xyzw[i];
				weights_offset += weight_elem_count_striped_per_entry * entry_id;
				int weight_count_per_feature_map_pair = window_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					weight_count_per_feature_map_pair *= window_sizes[i];

				#pragma unroll
				for(int i = 0; i < 2; ++i)
				{
					if (output_feature_map_id + i < output_feature_map_count)
					{
						int weights_offset1 = weights_offset + i * (input_feature_map_count_striped * weight_count_per_feature_map_pair);
						#pragma unroll
						for(int k = 0; k < WINDOW_WIDTH_LOCAL; ++k)
						{
							if (k < window_sizes[0] - weight_xyzw[0])
							{
								int weights_offset3 = weights_offset1 + k;
								float2 lr = learning_rate[weights_offset3];
								if (single_elem_per_destination)
								{
									float2 current_w = weights[weights_offset3];
									float new_val1 = current_w.x + lr.x * (sums[k * 4 + i * 2] - weight_decay * current_w.x);
									float new_val2 = current_w.y + lr.y * (sums[k * 4 + i * 2 + 1] - weight_decay * current_w.y);
									weights[weights_offset3] = make_float2(new_val1, new_val2);
								}
								else
								{
									float upd_val1 = lr.x * sums[k * 4 + i * 2];
									float upd_val2 = lr.y * sums[k * 4 + i * 2 + 1];
									atomicAdd((float *)(weights + weights_offset3), upd_val1);
									atomicAdd((float *)(weights + weights_offset3) + 1, upd_val2);
								}
							}
						}
					}
				}
			}
		}

#define launch_exact_kernel_const_const_const_const(dimension_count_const, window_width_const, block_size_const, single_input_feature_map_group_count_const) \
	convolution_tex_exact_blocked_upd_kernel_fermi<dimension_count_const,window_width_const,block_size_const,single_input_feature_map_group_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, *data[0], *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, forward_packed_config_count, forward_input_feature_map_group_size, different_input, weight_elem_count_striped_per_entry);

#define launch_generic_kernel_const_const_const(dimension_count_const, block_size_const, single_input_feature_map_group_count_const) \
	convolution_tex_generic_blocked_upd_kernel_fermi<dimension_count_const,block_size_const,single_input_feature_map_group_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, *data[0], *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, forward_packed_config_count, forward_input_feature_map_group_size, different_input, weight_elem_count_striped_per_entry);

#define launch_kernel_const_const_cost(dimension_count_const, window_width, block_size_const, single_input_feature_map_group_count_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 1, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 2: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 2, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 3: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 3, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 4: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 4, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 5: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 5, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 6: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 6, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 7: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 7, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 8: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 8, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 9: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 9, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		case 10: \
			launch_exact_kernel_const_const_const_const(dimension_count_const, 10, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		default: \
			launch_generic_kernel_const_const_const(dimension_count_const, block_size_const, single_input_feature_map_group_count_const); \
			break; \
		};

#define launch_kernel_const_const(dimension_count_const, window_width, block_size, single_input_feature_map_group_count_const) \
	switch (block_size) \
		{ \
		case 1: \
			launch_kernel_const_const_cost(dimension_count_const, window_width, 1, single_input_feature_map_group_count_const); \
			break; \
		case 2: \
			launch_kernel_const_const_cost(dimension_count_const, window_width, 2, single_input_feature_map_group_count_const); \
			break; \
		case 3: \
			launch_kernel_const_const_cost(dimension_count_const, window_width, 3, single_input_feature_map_group_count_const); \
			break; \
		case 4: \
			launch_kernel_const_const_cost(dimension_count_const, window_width, 4, single_input_feature_map_group_count_const); \
			break; \
		case 5: \
			launch_kernel_const_const_cost(dimension_count_const, window_width, 5, single_input_feature_map_group_count_const); \
			break; \
		};

#define launch_kernel(dimension_count_const, window_width, block_size, single_input_feature_map_group_count) \
	switch (single_input_feature_map_group_count) \
		{ \
		case false: \
			launch_kernel_const_const(dimension_count_const, window_width, block_size, false); \
			break; \
		case true: \
			launch_kernel_const_const(dimension_count_const, window_width, block_size, true); \
			break; \
		};

#define launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, window_width_const, block_size_const, single_output_feature_map_group_count_const) \
	convolution_backprop_tex_exact_blocked_upd_kernel_fermi<dimension_count_const,window_width_const,block_size_const,single_output_feature_map_group_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *data[0], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, output_configuration_specific_striped.feature_map_count, entry_count, backward_packed_config_count, backward_output_feature_map_group_size, weight_elem_count_striped_per_entry);

#define launch_backprop_generic_kernel_const_const_const(dimension_count_const, block_size_const, single_output_feature_map_group_count_const) \
	convolution_backprop_tex_generic_blocked_upd_kernel_fermi<dimension_count_const,block_size_const,single_output_feature_map_group_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *data[0], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, output_configuration_specific_striped.feature_map_count, entry_count, backward_packed_config_count, backward_output_feature_map_group_size, weight_elem_count_striped_per_entry);

#define launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, block_size_const, single_output_feature_map_group_count_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 1, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 2, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 3, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 4, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 5, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 6: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 6, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 7: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 7, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 8: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 8, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 9: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 9, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		case 10: \
			launch_backprop_exact_kernel_const_const_const_const(dimension_count_const, 10, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		default: \
			launch_backprop_generic_kernel_const_const_const(dimension_count_const, block_size_const, single_output_feature_map_group_count_const); \
			break; \
		};

#define launch_backprop_kernel_const_const(dimension_count_const, window_width, block_size, single_output_feature_map_group_count_const) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, 1, single_output_feature_map_group_count_const); \
			break; \
		case 2: \
			launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, 2, single_output_feature_map_group_count_const); \
			break; \
		case 3: \
			launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, 3, single_output_feature_map_group_count_const); \
			break; \
		case 4: \
			launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, 4, single_output_feature_map_group_count_const); \
			break; \
		case 5: \
			launch_backprop_kernel_const_const_cost(dimension_count_const, window_width, 5, single_output_feature_map_group_count_const); \
			break; \
		};

#define launch_backprop_kernel(dimension_count_const, window_width, block_size, single_output_feature_map_group_count) \
	switch (single_output_feature_map_group_count) \
		{ \
		case false: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, block_size, false); \
			break; \
		case true: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, block_size, true); \
			break; \
		};


#define launch_update_exact_kernel_const_const_const(dimension_count_const, window_width_const, single_elem_per_destination_const) \
	convolution_update_weights_exact_upd_kernel_fermi<dimension_count_const,window_width_const,single_elem_per_destination_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*data[0], output_errors, *learning_rate[0], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific_striped.feature_map_count, input_elem_count_per_entry_striped, output_elem_count_per_entry_striped, entry_count, different_input, updater_packed_config_count, updater_last_dimension_group_count, weight_elem_count_striped_per_entry, weight_decay);

#define launch_update_generic_kernel_const(dimension_count_const, single_elem_per_destination_const) \
	convolution_update_weights_generic_upd_kernel_fermi<dimension_count_const,single_elem_per_destination_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*data[0], output_errors, *learning_rate[0], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific_striped.feature_map_count, input_elem_count_per_entry_striped, output_elem_count_per_entry_striped, entry_count, different_input, updater_packed_config_count, updater_last_dimension_group_count, weight_elem_count_striped_per_entry, weight_decay);

#define launch_update_kernel_const_const(dimension_count_const, window_width, single_elem_per_destination_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 1, single_elem_per_destination_const); \
			break; \
		case 2: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 2, single_elem_per_destination_const); \
			break; \
		case 3: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 3, single_elem_per_destination_const); \
			break; \
		case 4: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 4, single_elem_per_destination_const); \
			break; \
		case 5: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 5, single_elem_per_destination_const); \
			break; \
		case 6: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 6, single_elem_per_destination_const); \
			break; \
		case 7: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 7, single_elem_per_destination_const); \
			break; \
		case 8: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 8, single_elem_per_destination_const); \
			break; \
		case 9: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 9, single_elem_per_destination_const); \
			break; \
		case 10: \
			launch_update_exact_kernel_const_const_const(dimension_count_const, 10, single_elem_per_destination_const); \
			break; \
		default: \
			launch_update_generic_kernel_const(dimension_count_const, single_elem_per_destination_const); \
			break; \
		};

#define launch_update_kernel(dimension_count_const, window_width, single_elem_per_destination) \
	switch (single_elem_per_destination) \
		{ \
		case false: \
			launch_update_kernel_const_const(dimension_count_const, window_width, false); \
			break; \
		case true: \
			launch_update_kernel_const_const(dimension_count_const, window_width, true); \
			break; \
		}

		template<int dimension_count>
		class convolution_layer_updater_cuda_fermi : public layer_updater_cuda
		{
		public:
			convolution_layer_updater_cuda_fermi()
			{
				input_tex_ref.addressMode[0] = cudaAddressModeBorder;
				input_tex_ref.normalized = false;
				output_tex_ref.addressMode[0] = cudaAddressModeBorder;
				output_tex_ref.normalized = false;
			}

			virtual ~convolution_layer_updater_cuda_fermi()
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
				cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
				cuda_safe_call(cudaBindTexture(0, input_tex_ref, *additional_buffers[0], desc, additional_buffers[0]->get_size()));

				int original_input_elem_offset = offset_input_entry_id * input_elem_count_per_entry;
				cuda_util::copy_to_striped(
					*cuda_config,
					(const float *)(*input_neurons_buffer) + original_input_elem_offset,
					*additional_buffers[0],
					input_elem_count_per_feature_map,
					input_configuration_specific.feature_map_count,
					different_input ? entry_count : 1,
					stream_id);

				if (forward_input_feature_map_group_count > 1)
					cuda_util::set_with_value(
						*cuda_config,
						*output_neurons_buffer,
						0.0F,
						output_elem_count_per_entry * entry_count,
						stream_id);

				const packed_config<forward_dimension_count> * packed_config_list = static_cast<const packed_config<forward_dimension_count> *>((const void *)*additional_buffers[2]);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_packed_config_count,
					entry_count,
					1);

				bool single_input_feature_map_group_count = (forward_input_feature_map_group_count == 1);

				launch_kernel(dimension_count, window_sizes[0], forward_x_block_size, single_input_feature_map_group_count);
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
				if (!different_input)
					throw neural_network_exception("convolution_layer_updater_cuda_fermi is not able to backprop to the same input");

				if (!backprop_required)
					throw neural_network_exception("convolution_layer_updater_cuda_fermi is not configured to do backprop but requested to");

				cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
				cuda_safe_call(cudaBindTexture(0, output_tex_ref, *additional_buffers[1], desc, additional_buffers[1]->get_size()));

				cuda_util::copy_to_striped(
					*cuda_config,
					*output_errors_buffer,
					*additional_buffers[1],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				if (backward_output_feature_map_group_count > 1)
					cuda_util::set_with_value(
						*cuda_config,
						*input_errors_buffer,
						0.0F,
						input_elem_count_per_entry * entry_count,
						stream_id);

				const packed_config<backward_dimension_count> * packed_config_list = static_cast<const packed_config<backward_dimension_count> *>((const void *)*additional_buffers[4]);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					backward_packed_config_count,
					entry_count,
					1);

				bool single_output_feature_map_group_count = (backward_output_feature_map_group_count == 1);

				launch_backprop_kernel(dimension_count, window_sizes[0], backward_x_block_size, single_output_feature_map_group_count);
			}

			virtual void enqueue_update_weights(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& learning_rate,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count,
				float weight_decay)
			{
				// Update biases
				{
					int threadblock_size = get_threadblock_size_biases(output_elem_count_per_feature_map);
					dim3 grid_size(1, output_configuration_specific.feature_map_count, entry_count);
					dim3 block_size(threadblock_size, 1, 1);
					int smem_size = threadblock_size * sizeof(float);
					int min_iteration_count = output_elem_count_per_feature_map / threadblock_size;

					convolution_update_biases_upd_kernel_fermi<<<grid_size, block_size, smem_size, stream_id>>>(
						*data[1],
						*output_errors_buffer,
						*learning_rate[1],
						output_configuration_specific.feature_map_count,
						output_elem_count_per_feature_map,
						min_iteration_count);
				}

				if (!backprop_required)
				{
					cuda_util::copy_to_striped(
						*cuda_config,
						*output_errors_buffer,
						*additional_buffers[1],
						output_elem_count_per_feature_map,
						output_configuration_specific.feature_map_count,
						entry_count,
						stream_id);
				}
				const float2 * output_errors = *additional_buffers[1];

				cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
				cuda_safe_call(cudaBindTexture(0, input_tex_ref, *additional_buffers[0], desc, additional_buffers[0]->get_size()));

				const packed_config<updater_dimension_count> * packed_config_list = static_cast<const packed_config<updater_dimension_count> *>((const void *)*additional_buffers[3]);

				if (!updater_single_elem_per_destination)
				{
					cuda_util::apply_weight_decay(
						*cuda_config,
						*learning_rate[0],
						*data[0],
						weight_decay,
						entry_count * weight_elem_count_striped_per_entry * 2,
						stream_id);
				}

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					updater_packed_config_count,
					entry_count,
					1);

				launch_update_kernel(dimension_count, window_sizes[0], updater_single_elem_per_destination);
			}

		protected:
			static const int forward_dimension_count = (dimension_count + 2);
			static const int backward_dimension_count = (dimension_count + 2);
			static const int updater_dimension_count = (dimension_count * 2 + 2);

			virtual bool is_in_place_backprop() const
			{
				return false;
			}

			virtual void updater_configured()
			{
				nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

				for(int i = 0; i < dimension_count; ++i)
				{
					window_sizes[i] = layer_derived->window_sizes[i];
					input_sizes[i] = input_configuration_specific.dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];
				}

				{
					input_configuration_specific_striped = cuda_util::get_layer_configuration_specific_striped(input_configuration_specific);
					input_elem_count_per_entry_striped = input_configuration_specific_striped.get_neuron_count();

					forward_x_block_size = get_block_size(output_configuration_specific.dimension_sizes[0]);
					forward_x_block_count = (output_configuration_specific.dimension_sizes[0] + forward_x_block_size - 1) / forward_x_block_size;
					forward_output_feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

					forward_packed_config_count = forward_x_block_count * input_configuration_specific_striped.feature_map_count * forward_output_feature_map_block_count;
					for(int i = 1; i < dimension_count; ++i)
						forward_packed_config_count *= output_sizes[i];
				}

				{
					output_configuration_specific_striped = cuda_util::get_layer_configuration_specific_striped(output_configuration_specific);
					output_elem_count_per_entry_striped = output_configuration_specific_striped.get_neuron_count();

					backward_x_block_size = get_block_size(input_configuration_specific.dimension_sizes[0]);
					backward_x_block_count = (input_configuration_specific.dimension_sizes[0] + backward_x_block_size - 1) / backward_x_block_size;
					backward_input_feature_map_block_count = (input_configuration_specific_striped.feature_map_count + (FEATURE_MAP_BLOCK_SIZE/2) - 1) / (FEATURE_MAP_BLOCK_SIZE/2);

					backward_packed_config_count = backward_x_block_count * backward_input_feature_map_block_count * output_configuration_specific_striped.feature_map_count;
					for(int i = 1; i < dimension_count; ++i)
						backward_packed_config_count *= input_sizes[i];
				}

				{
					updater_window_x_block_count = (window_sizes[0] <= MAX_WINDOW_WIDTH) ? 1 : (window_sizes[0] + WINDOW_WIDTH_LOCAL - 1) / WINDOW_WIDTH_LOCAL;
					updater_packed_config_count = output_configuration_specific_striped.feature_map_count * input_configuration_specific_striped.feature_map_count * updater_window_x_block_count;
					for(int i = 1; i < dimension_count; ++i)
					{
						updater_packed_config_count *= window_sizes[i];
						updater_packed_config_count *= output_sizes[i];
					}
					updater_last_dimension_group_count = (dimension_count > 1) ? output_sizes[dimension_count - 1] : 1;
				}

				{
					weight_elem_count_striped_per_entry = ((output_configuration_specific.feature_map_count + 1) & ~1) * input_configuration_specific_striped.feature_map_count;
					for(int i = 0; i < dimension_count; ++i)
						weight_elem_count_striped_per_entry *= window_sizes[i];
				}
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const
			{
				std::vector<size_t> res;

				res.push_back(input_elem_count_per_entry_striped * sizeof(float2));
				res.push_back(output_elem_count_per_entry_striped * sizeof(float2));

				return res;
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const
			{
				std::vector<size_t> res;

				res.push_back(sizeof(packed_config<forward_dimension_count>) * forward_packed_config_count);

				res.push_back(sizeof(packed_config<updater_dimension_count>) * updater_packed_config_count);

				if (backprop_required)
					res.push_back(sizeof(packed_config<backward_dimension_count>) * backward_packed_config_count);

				return res;
			}

			virtual void set_max_entry_count(unsigned int max_entry_count)
			{
				{
					forward_packed_config_count = forward_x_block_count * forward_output_feature_map_block_count;
					for(int i = 1; i < dimension_count; ++i)
						forward_packed_config_count *= output_sizes[i];
					forward_input_feature_map_group_count = cuda_util::get_group_count(
						*cuda_config,
						forward_packed_config_count * max_entry_count,
						input_configuration_specific_striped.feature_map_count);
					forward_input_feature_map_group_size = (input_configuration_specific_striped.feature_map_count + forward_input_feature_map_group_count - 1) / forward_input_feature_map_group_count;
					forward_packed_config_count *= forward_input_feature_map_group_count;
				}

				{
					updater_packed_config_count = output_configuration_specific_striped.feature_map_count * input_configuration_specific_striped.feature_map_count * updater_window_x_block_count;
					for(int i = 1; i < dimension_count; ++i)
					{
						updater_packed_config_count *= window_sizes[i];
						updater_packed_config_count *= (i == dimension_count - 1) ? 1 : output_sizes[i];
					}
					if (dimension_count > 1)
					{
						updater_last_dimension_group_count = cuda_util::get_group_count(
							*cuda_config,
							updater_packed_config_count * max_entry_count,
							output_sizes[dimension_count - 1]);
						updater_packed_config_count *= updater_last_dimension_group_count;
					}
					else
						updater_last_dimension_group_count = 1;

					updater_single_elem_per_destination = (updater_window_x_block_count == 1) && (updater_last_dimension_group_count == 1);
					for(int i = 1; i < dimension_count - 1; ++i)
						updater_single_elem_per_destination = updater_single_elem_per_destination && (output_sizes[i] == 1);
				}

				if (backprop_required)
				{
					backward_packed_config_count = backward_x_block_count * backward_input_feature_map_block_count;
					for(int i = 1; i < dimension_count; ++i)
						backward_packed_config_count *= input_sizes[i];
					backward_output_feature_map_group_count = cuda_util::get_group_count(
						*cuda_config,
						backward_packed_config_count * max_entry_count,
						output_configuration_specific_striped.feature_map_count);
					backward_output_feature_map_group_size = (output_configuration_specific_striped.feature_map_count + backward_output_feature_map_group_count - 1) / backward_output_feature_map_group_count;
					backward_packed_config_count *= backward_output_feature_map_group_count;
				}
			}

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const
			{
				std::vector<unsigned int> res;

				res.push_back(input_elem_count_per_entry_striped);
				res.push_back(output_elem_count_per_entry_striped);

				return res;
			}

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
			{
				{
					std::vector<packed_config<forward_dimension_count> > task_list;
					{
						nnforge_array<int, dimension_count> size_list;
						for(int i = 0; i < dimension_count; ++i)
							size_list[i] = (i == 0) ? forward_x_block_count : output_sizes[i];
						std::vector<nnforge_array<int, dimension_count> > ordered_list;
						sequential_curve<dimension_count>::fill_pattern(size_list, ordered_list);
						packed_config<forward_dimension_count> new_elem;
						for(int input_feature_map_group_id = 0; input_feature_map_group_id < forward_input_feature_map_group_count; ++input_feature_map_group_id)
						{
							new_elem.set_val(dimension_count + 1, input_feature_map_group_id * forward_input_feature_map_group_size);
							for(int output_feature_map_block_id = 0; output_feature_map_block_id < forward_output_feature_map_block_count; ++output_feature_map_block_id)
							{
								new_elem.set_val(dimension_count, output_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
								for(int j = 0; j < ordered_list.size(); ++j)
								{
									const nnforge_array<int, dimension_count>& spatial_dimensions = ordered_list[j];
									for(int i = 0; i < dimension_count; ++i)
										new_elem.set_val(i, (i == 0) ? (spatial_dimensions[i] * forward_x_block_size) : spatial_dimensions[i]);
									task_list.push_back(new_elem);
								}
							}
						}
					}
					cuda_safe_call(cudaMemcpy(*additional_buffers[2], &(*task_list.begin()), sizeof(packed_config<forward_dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
				}

				{
					std::vector<packed_config<updater_dimension_count> > task_list;

					nnforge_array<int, dimension_count * 2> size_list;
					for(int i = 1; i < dimension_count; ++i)
					{
						size_list[i - 1] = window_sizes[i];
						size_list[(dimension_count - 1) + i - 1] = ((dimension_count > 1) && (i == dimension_count - 1)) ? updater_last_dimension_group_count : output_sizes[i];
					}
					size_list[(dimension_count - 1) * 2] = input_configuration_specific_striped.feature_map_count;
					size_list[(dimension_count - 1) * 2 + 1] = output_configuration_specific_striped.feature_map_count;
					std::vector<nnforge_array<int, dimension_count*2> > updater_config_ordered_list;
					space_filling_curve<dimension_count*2>::fill_pattern(size_list, updater_config_ordered_list);

					packed_config<updater_dimension_count> new_elem;
					new_elem.set_val(dimension_count, 0);
					for(int ordered_elem_id = 0; ordered_elem_id < updater_config_ordered_list.size(); ++ordered_elem_id)
					{
						const nnforge_array<int, dimension_count*2>& ordered_elem = updater_config_ordered_list[ordered_elem_id];
						for(int i = 1; i < dimension_count; ++i)
						{
							new_elem.set_val(i, ordered_elem[i - 1]);
							new_elem.set_val(dimension_count + i, ordered_elem[(dimension_count - 1) + i - 1]);
						}
						new_elem.set_val(dimension_count * 2, ordered_elem[(dimension_count - 1) * 2]);
						new_elem.set_val(dimension_count * 2 + 1, ordered_elem[(dimension_count - 1) * 2 + 1]);

						for(int i = 0; i < updater_window_x_block_count; ++i)
						{
							new_elem.set_val(0, i * WINDOW_WIDTH_LOCAL);
							task_list.push_back(new_elem);
						}
					}

					cuda_safe_call(cudaMemcpy(*additional_buffers[3], &(*task_list.begin()), sizeof(packed_config<updater_dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
				}

				if (backprop_required)
				{
					std::vector<packed_config<backward_dimension_count> > task_list;
					{
						nnforge_array<int, dimension_count> size_list;
						for(int i = 0; i < dimension_count; ++i)
							size_list[i] = (i == 0) ? backward_x_block_count : input_sizes[i];
						std::vector<nnforge_array<int, dimension_count> > ordered_list;
						sequential_curve<dimension_count>::fill_pattern(size_list, ordered_list);
						packed_config<backward_dimension_count> new_elem;
						for(int output_feature_map_group_id = 0; output_feature_map_group_id < backward_output_feature_map_group_count; ++output_feature_map_group_id)
						{
							new_elem.set_val(dimension_count + 1, output_feature_map_group_id * backward_output_feature_map_group_size);
							for(int input_feature_map_block_id = 0; input_feature_map_block_id < backward_input_feature_map_block_count; ++input_feature_map_block_id)
							{
								new_elem.set_val(dimension_count, input_feature_map_block_id * (FEATURE_MAP_BLOCK_SIZE/2));
								for(int j = 0; j < ordered_list.size(); ++j)
								{
									const nnforge_array<int, dimension_count>& spatial_dimensions = ordered_list[j];
									for(int i = 0; i < dimension_count; ++i)
										new_elem.set_val(i, (i == 0) ? (spatial_dimensions[i] * backward_x_block_size + backward_x_block_size - 1) : spatial_dimensions[i]);
									task_list.push_back(new_elem);
								}
							}
						}
					}
					cuda_safe_call(cudaMemcpy(*additional_buffers[4], &(*task_list.begin()), sizeof(packed_config<backward_dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
				}
			}

			virtual unsigned int get_data_elem_count(unsigned int part_id, unsigned int source_elem_count) const
			{
				if (part_id != 0)
					return layer_updater_cuda::get_data_elem_count(part_id, source_elem_count);

				return weight_elem_count_striped_per_entry * 2;
			}

			virtual std::vector<unsigned int> get_incoming_weight_count_per_output_neuron_list() const
			{
				std::vector<unsigned int> res;

				unsigned int weight_elem_count = input_configuration_specific_striped.feature_map_count * 2;
				for(int i = 0; i < dimension_count; ++i)
					weight_elem_count *= window_sizes[i];

				res.push_back(weight_elem_count);
				res.push_back(1);

				return res;
			}

			virtual void fill_data_for_device(
				unsigned int part_id,
				const float * src,
				float * dst,
				unsigned int count) const
			{
				if (part_id != 0)
					return layer_updater_cuda::fill_data_for_device(part_id, src, dst, count);

				unsigned int window_total_size = 1;
				for(int i = 0; i < dimension_count; ++i)
					window_total_size *= window_sizes[i];

				if (output_configuration_specific.feature_map_count & 1)
				{
					unsigned int zero_elem_count = window_total_size * input_configuration_specific_striped.feature_map_count * 2;
					std::fill_n(
						dst + weight_elem_count_striped_per_entry * 2 - zero_elem_count,
						zero_elem_count,
						0.0F);
				}

				unsigned int input_feature_map_count_striped = input_configuration_specific_striped.feature_map_count;

				unsigned int src_offset = 0;
				unsigned int dst_offset = 0;
				for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_configuration_specific.feature_map_count; ++output_feature_map_id)
				{
					for(unsigned int input_feature_map_id_striped = 0; input_feature_map_id_striped < input_feature_map_count_striped; ++input_feature_map_id_striped, dst_offset += window_total_size * 2)
					{
						bool second_feature_map_present = (input_feature_map_id_striped * 2 + 1 < input_configuration_specific.feature_map_count);
						for(int dst_elem_id = 0; dst_elem_id < window_total_size; ++dst_elem_id)
						{
							dst[dst_offset + dst_elem_id * 2] = src[src_offset + dst_elem_id];
							float other_val = 0.0F;
							if (second_feature_map_present)
								other_val = src[src_offset + dst_elem_id + window_total_size];
							dst[dst_offset + dst_elem_id * 2 + 1] = other_val;
						}

						src_offset += window_total_size * (second_feature_map_present ? 2 : 1);
					}
				}
			}

			virtual void fill_data_for_host(
				unsigned int part_id,
				const float * src,
				float * dst,
				unsigned int count) const
			{
				if (part_id != 0)
					return layer_updater_cuda::fill_data_for_host(part_id, src, dst, count);

				unsigned int window_total_size = 1;
				for(int i = 0; i < dimension_count; ++i)
					window_total_size *= window_sizes[i];

				unsigned int input_feature_map_count_striped = input_configuration_specific_striped.feature_map_count;

				unsigned int src_offset = 0;
				unsigned int dst_offset = 0;
				for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_configuration_specific.feature_map_count; ++output_feature_map_id)
				{
					for(unsigned int input_feature_map_id_striped = 0; input_feature_map_id_striped < input_feature_map_count_striped; ++input_feature_map_id_striped, src_offset += window_total_size * 2)
					{
						bool second_feature_map_present = (input_feature_map_id_striped * 2 + 1 < input_configuration_specific.feature_map_count);
						for(int src_elem_id = 0; src_elem_id < window_total_size; ++src_elem_id)
						{
							dst[dst_offset + src_elem_id] = src[src_offset + src_elem_id * 2];
							if (second_feature_map_present)
								dst[dst_offset + src_elem_id + window_total_size] = src[src_offset + src_elem_id * 2 + 1];
						}

						dst_offset += window_total_size * (second_feature_map_present ? 2 : 1);
					}
				}
			}

			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> window_sizes;

			layer_configuration_specific input_configuration_specific_striped;
			layer_configuration_specific output_configuration_specific_striped;
			unsigned int input_elem_count_per_entry_striped;
			unsigned int output_elem_count_per_entry_striped;

			int forward_x_block_size;
			int forward_x_block_count;
			int forward_input_feature_map_group_count;
			int forward_input_feature_map_group_size;
			int forward_output_feature_map_block_count;
			int forward_packed_config_count;

			int backward_x_block_size;
			int backward_x_block_count;
			int backward_output_feature_map_group_count;
			int backward_output_feature_map_group_size;
			int backward_input_feature_map_block_count;
			int backward_packed_config_count;

			int updater_packed_config_count;
			int updater_window_x_block_count;
			int updater_last_dimension_group_count;
			bool updater_single_elem_per_destination;

			unsigned int weight_elem_count_striped_per_entry;

		private:
			static int get_block_size(int width)
			{
				int block_count = (width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
				int block_size = (width + block_count - 1) / block_count;
				return block_size;
			}

			static int get_threadblock_size_biases(int output_neuron_count)
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
		};
	}
}
