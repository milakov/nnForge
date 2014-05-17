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

#include "layer_hessian_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "cuda_texture.h"
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
		template<int DIMENSION_COUNT, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_tex_generic_blocked_hess_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int xyzw[DIMENSION_COUNT];
			xyzw[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw[0] < output_sizes[0]) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = input_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_output_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
					xyzw[i + 1] = conf.get_val(i);
				int output_feature_map_id = conf.get_val(DIMENSION_COUNT - 1);
				int input_elem_id = entry_id * input_feature_map_count_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i];
				int weights_offset = weight_count_per_output_feature_map * output_feature_map_id;

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

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count_striped; ++input_layer_id)
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
										weight_list[i] = tex1Dfetch<float2>(weights_tex, weights_offset + weight_count_per_output_feature_map * i);
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										float2 inp = tex1Dfetch<float2>(input_tex, input_elem_id + j); 
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									weights_offset++;
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
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_tex_exact_blocked_hess_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int xyzw[DIMENSION_COUNT];
			xyzw[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw[0] < output_sizes[0]) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = input_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_output_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
					xyzw[i + 1] = conf.get_val(i);
				int output_feature_map_id = conf.get_val(DIMENSION_COUNT - 1);
				int input_elem_id = entry_id * input_feature_map_count_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i];
				int weights_offset = weight_count_per_output_feature_map * output_feature_map_id;

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

				for(int input_layer_id = 0; input_layer_id < input_feature_map_count_striped; ++input_layer_id)
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
										weight_list[i] = tex1Dfetch<float2>(weights_tex, weights_offset + weight_count_per_output_feature_map * i);
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										float2 inp = tex1Dfetch<float2>(input_tex, input_elem_id + j); 
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									weights_offset++;
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
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_bbprop_tex_exact_blocked_hess_kernel_kepler(
			float * __restrict input_errors,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t weights_squared_tex,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int output_feature_map_count_striped,
			int entry_count,
			int packed_config_count)
		{
			int xyzw[DIMENSION_COUNT];
			xyzw[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE + (BLOCK_SIZE - 1);
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw[0] < input_sizes[0] + (BLOCK_SIZE - 1)) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_input_feature_map = output_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_input_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
					xyzw[i + 1] = conf.get_val(i);
				int input_feature_map_id = conf.get_val(DIMENSION_COUNT - 1);
				int output_elem_id = entry_id * output_feature_map_count_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw[i];
				int weights_offset = weight_count_per_input_feature_map * input_feature_map_id;

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int min_exclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					min_exclusive[i] = xyzw[i] - output_sizes[i];
				int max_inclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					max_inclusive[i] = xyzw[i];

				for(int output_layer_id = 0; output_layer_id < output_feature_map_count_striped; ++output_layer_id)
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
									output_vals[i] = tex1Dfetch<float2>(output_tex, b_fit0 ? (output_elem_id - i) : -1);
								}

								#pragma unroll
								for(int input_x = 0; input_x < WINDOW_WIDTH; ++input_x)
								{
									float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										weight_list[i] = tex1Dfetch<float2>(weights_squared_tex, weights_offset + weight_count_per_input_feature_map * i);

									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += output_vals[input_x + j].x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += output_vals[input_x + j].y * weight_list[i].y;
										}
									}
									weights_offset++;
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
				} // for(int output_layer_id

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
								*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_bbprop_tex_generic_blocked_hess_kernel_kepler(
			float * __restrict input_errors,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t weights_squared_tex,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			int input_feature_map_count,
			int output_feature_map_count_striped,
			int entry_count,
			int packed_config_count)
		{
			int xyzw[DIMENSION_COUNT];
			xyzw[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE + (BLOCK_SIZE - 1);
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw[0] < input_sizes[0] + (BLOCK_SIZE - 1)) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_input_feature_map = output_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_input_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
					xyzw[i + 1] = conf.get_val(i);
				int input_feature_map_id = conf.get_val(DIMENSION_COUNT - 1);
				int output_elem_id = entry_id * output_feature_map_count_striped;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw[i];
				int weights_offset = weight_count_per_input_feature_map * input_feature_map_id;

				float sums[FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE];
				#pragma unroll
				for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE * BLOCK_SIZE; ++i)
					sums[i] = 0.0F;

				int min_exclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					min_exclusive[i] = xyzw[i] - output_sizes[i];
				int max_inclusive[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					max_inclusive[i] = xyzw[i];

				for(int output_layer_id = 0; output_layer_id < output_feature_map_count_striped; ++output_layer_id)
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
										bool b_fit0 = b_fit1 && (input_x + i > min_exclusive[0]) && (input_x + i <= max_inclusive[0]);
										output_vals[i] = tex1Dfetch<float2>(output_tex, b_fit0 ? (output_elem_id - i) : -1);
									}
									output_elem_id -= WINDOW_WIDTH_LOCAL;

									#pragma unroll
									for(int input_x_local = 0; input_x_local < WINDOW_WIDTH_LOCAL; ++input_x_local)
									{
										float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											weight_list[i] = tex1Dfetch<float2>(weights_squared_tex, weights_offset + weight_count_per_input_feature_map * i);

										#pragma unroll
										for(int j = 0; j < BLOCK_SIZE; ++j)
										{
											#pragma unroll
											for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
											{
												sums[i * BLOCK_SIZE + j] += output_vals[input_x_local + j].x * weight_list[i].x;
												sums[i * BLOCK_SIZE + j] += output_vals[input_x_local + j].y * weight_list[i].y;
											}
										}
										weights_offset++;
									}
								}
								#pragma unroll 1
								for(; input_x < window_sizes[0]; ++input_x)
								{
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										bool b_fit0 = b_fit1 && (input_x + j > min_exclusive[0]) && (input_x + j <= max_inclusive[0]);
										float2 inp = tex1Dfetch<float2>(output_tex, b_fit0 ? (output_elem_id - j) : -1);
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											float2 w = tex1Dfetch<float2>(weights_squared_tex, weights_offset + weight_count_per_input_feature_map * i);
											sums[i * BLOCK_SIZE + j] += inp.x * w.x;
											sums[i * BLOCK_SIZE + j] += inp.y * w.y;
										}
									}
									weights_offset++;
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
				} // for(int output_layer_id

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
								*(base_input + input_neuron_count_per_feature_map * i - j) = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		extern __shared__ float arr[];
		__global__ void convolution_update_biases_hess_kernel_kepler(
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

		template<int DIMENSION_COUNT, int WINDOW_WIDTH>
		__launch_bounds__(256, 4)
		__global__ void convolution_update_weights_exact_hess_kernel_kepler(
			float * __restrict hessian_weights,
			cudaTextureObject_t input_squared_tex,
			cudaTextureObject_t output_tex,
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
			int block_size,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_entry_id = (blockIdx.y * blockDim.y + threadIdx.y) * block_size;

			bool in_bounds = (packed_config_id < packed_config_count) && (base_entry_id < entry_count);
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

				int iteration_count = min(block_size, entry_count - base_entry_id);

				int output_errors_offset = base_entry_id * output_feature_map_count_striped + output_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_errors_offset = output_errors_offset * output_sizes[i] + xyzw[i];

				int input_elem_id = base_entry_id * input_feature_map_count_striped + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i] + weight_xyzw[i];

				float sums[WINDOW_WIDTH * 4];
				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH * 4; ++i)
					sums[i] = 0.0F;

				int iteration_count_x = output_sizes[0];

				int output_shift = output_elem_count_per_entry_striped - iteration_count_x;
				int input_shift = input_elem_count_per_entry_striped - iteration_count_x - (WINDOW_WIDTH - 1);

				for(int t = 0; t < iteration_count; ++t)
				{
					float2 input_squared_buf[WINDOW_WIDTH];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH; ++i)
					{
						input_squared_buf[i] = tex1Dfetch<float2>(input_squared_tex, input_elem_id);
						++input_elem_id;
					}

					#pragma unroll 4
					for(int x = 0; x < iteration_count_x; ++x)
					{
						float2 output_error = tex1Dfetch<float2>(output_tex, output_errors_offset);

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH - 1; ++i)
							input_squared_buf[i] = input_squared_buf[i + 1];
						input_squared_buf[WINDOW_WIDTH - 1] = tex1Dfetch<float2>(input_squared_tex, input_elem_id);

						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH; ++j)
						{
							sums[j * 4] += output_error.x * input_squared_buf[j].x;
							sums[j * 4 + 1] += output_error.x * input_squared_buf[j].y;
							sums[j * 4 + 2] += output_error.y * input_squared_buf[j].x;
							sums[j * 4 + 3] += output_error.y * input_squared_buf[j].y;
						}

						output_errors_offset++;
						input_elem_id++;
					}

					output_errors_offset += output_shift;
					input_elem_id += input_shift;
				}

				int output_feature_map_id = output_feature_map_striped_id * 2;
				int input_feature_map_id = input_feature_map_striped_id * 2;
				int weights_offset = output_feature_map_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					weights_offset = weights_offset * window_sizes[i] + weight_xyzw[i];
				int weight_count_per_feature_map_pair = window_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					weight_count_per_feature_map_pair *= window_sizes[i];

				#pragma unroll
				for(int i = 0; i < 2; ++i)
				{
					if (output_feature_map_id + i < output_feature_map_count)
					{
						int weights_offset1 = weights_offset + i * (input_feature_map_count * weight_count_per_feature_map_pair);
						#pragma unroll
						for(int j = 0; j < 2; ++j)
						{
							if (input_feature_map_id + j < input_feature_map_count)
							{
								int weights_offset2 = weights_offset1 + j * weight_count_per_feature_map_pair;
								#pragma unroll
								for(int k = 0; k < WINDOW_WIDTH; ++k)
								{
									int weights_offset3 = weights_offset2 + k;
									atomicAdd(hessian_weights + weights_offset3, sums[k * 4 + i * 2 + j]);
								}
							}
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT>
		__launch_bounds__(256, 4)
		__global__ void convolution_update_weights_generic_hess_kernel_kepler(
			float * __restrict hessian_weights,
			cudaTextureObject_t input_squared_tex,
			cudaTextureObject_t output_tex,
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
			int block_size,
			int packed_config_count)
		{
			int packed_config_id = blockIdx.x * blockDim.x + threadIdx.x;
			int base_entry_id = (blockIdx.y * blockDim.y + threadIdx.y) * block_size;

			bool in_bounds = (packed_config_id < packed_config_count) && (base_entry_id < entry_count);
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

				int iteration_count = min(block_size, entry_count - base_entry_id);

				int output_errors_offset = base_entry_id * output_feature_map_count_striped + output_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_errors_offset = output_errors_offset * output_sizes[i] + xyzw[i];

				int input_elem_id = base_entry_id * input_feature_map_count_striped + input_feature_map_striped_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw[i] + weight_xyzw[i];

				float sums[WINDOW_WIDTH_LOCAL * 4];
				#pragma unroll
				for(int i = 0; i < WINDOW_WIDTH_LOCAL * 4; ++i)
					sums[i] = 0.0F;

				int iteration_count_x = output_sizes[0];

				int output_shift = output_elem_count_per_entry_striped - iteration_count_x;
				int input_shift = input_elem_count_per_entry_striped - iteration_count_x - (WINDOW_WIDTH_LOCAL - 1);

				for(int t = 0; t < iteration_count; ++t)
				{
					float2 input_squared_buf[WINDOW_WIDTH_LOCAL];
					#pragma unroll
					for(int i = 1; i < WINDOW_WIDTH_LOCAL; ++i)
					{
						input_squared_buf[i] = tex1Dfetch<float2>(input_squared_tex, input_elem_id);
						++input_elem_id;
					}

					#pragma unroll 4
					for(int x = 0; x < iteration_count_x; ++x)
					{
						float2 output_error = tex1Dfetch<float2>(output_tex, output_errors_offset);

						#pragma unroll
						for(int i = 0; i < WINDOW_WIDTH_LOCAL - 1; ++i)
							input_squared_buf[i] = input_squared_buf[i + 1];
						input_squared_buf[WINDOW_WIDTH_LOCAL - 1] = tex1Dfetch<float2>(input_squared_tex, input_elem_id);

						#pragma unroll
						for(int j = 0; j < WINDOW_WIDTH_LOCAL; ++j)
						{
							sums[j * 4] += output_error.x * input_squared_buf[j].x;
							sums[j * 4 + 1] += output_error.x * input_squared_buf[j].y;
							sums[j * 4 + 2] += output_error.y * input_squared_buf[j].x;
							sums[j * 4 + 3] += output_error.y * input_squared_buf[j].y;
						}

						output_errors_offset++;
						input_elem_id++;
					}

					output_errors_offset += output_shift;
					input_elem_id += input_shift;
				}

				int output_feature_map_id = output_feature_map_striped_id * 2;
				int input_feature_map_id = input_feature_map_striped_id * 2;
				int weights_offset = output_feature_map_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					weights_offset = weights_offset * window_sizes[i] + weight_xyzw[i];
				int weight_count_per_feature_map_pair = window_sizes[0];
				for(int i = 1; i < DIMENSION_COUNT; ++i)
					weight_count_per_feature_map_pair *= window_sizes[i];

				#pragma unroll
				for(int i = 0; i < 2; ++i)
				{
					if (output_feature_map_id + i < output_feature_map_count)
					{
						int weights_offset1 = weights_offset + i * (input_feature_map_count * weight_count_per_feature_map_pair);
						#pragma unroll
						for(int j = 0; j < 2; ++j)
						{
							if (input_feature_map_id + j < input_feature_map_count)
							{
								int weights_offset2 = weights_offset1 + j * weight_count_per_feature_map_pair;
								#pragma unroll
								for(int k = 0; k < WINDOW_WIDTH_LOCAL; ++k)
								{
									if (k < window_sizes[0] - weight_xyzw[0])
									{
										int weights_offset3 = weights_offset2 + k;
										atomicAdd(hessian_weights + weights_offset3, sums[k * 4 + i * 2 + j]);
									}
								}
							}
						}
					}
				}
			}
		}

#define launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, block_size_const) \
	convolution_tex_exact_blocked_hess_kernel_kepler<dimension_count_const,window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, forward_packed_config_count);

#define launch_generic_kernel_const_const(dimension_count_const, block_size_const) \
	convolution_tex_generic_blocked_hess_kernel_kepler<dimension_count_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, forward_packed_config_count);

#define launch_kernel_const_const(dimension_count_const, window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_exact_kernel_const_const_const(dimension_count_const, 1, block_size_const); \
			break; \
		case 2: \
			launch_exact_kernel_const_const_const(dimension_count_const, 2, block_size_const); \
			break; \
		case 3: \
			launch_exact_kernel_const_const_const(dimension_count_const, 3, block_size_const); \
			break; \
		case 4: \
			launch_exact_kernel_const_const_const(dimension_count_const, 4, block_size_const); \
			break; \
		case 5: \
			launch_exact_kernel_const_const_const(dimension_count_const, 5, block_size_const); \
			break; \
		case 6: \
			launch_exact_kernel_const_const_const(dimension_count_const, 6, block_size_const); \
			break; \
		case 7: \
			launch_exact_kernel_const_const_const(dimension_count_const, 7, block_size_const); \
			break; \
		case 8: \
			launch_exact_kernel_const_const_const(dimension_count_const, 8, block_size_const); \
			break; \
		case 9: \
			launch_exact_kernel_const_const_const(dimension_count_const, 9, block_size_const); \
			break; \
		case 10: \
			launch_exact_kernel_const_const_const(dimension_count_const, 10, block_size_const); \
			break; \
		default: \
			launch_generic_kernel_const_const(dimension_count_const, block_size_const); \
			break; \
		};

#define launch_kernel(dimension_count_const, window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_kernel_const_const(dimension_count_const, window_width, 1); \
			break; \
		case 2: \
			launch_kernel_const_const(dimension_count_const, window_width, 2); \
			break; \
		case 3: \
			launch_kernel_const_const(dimension_count_const, window_width, 3); \
			break; \
		case 4: \
			launch_kernel_const_const(dimension_count_const, window_width, 4); \
			break; \
		case 5: \
			launch_kernel_const_const(dimension_count_const, window_width, 5); \
			break; \
		};

#define launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, block_size_const) \
	convolution_bbprop_tex_exact_blocked_hess_kernel_kepler<dimension_count_const,window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, weights_squared_tex, packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific_striped.feature_map_count, entry_count, backward_packed_config_count);

#define launch_backprop_generic_kernel_const_const(dimension_count_const, block_size_const) \
	convolution_bbprop_tex_generic_blocked_hess_kernel_kepler<dimension_count_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, weights_squared_tex, packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific_striped.feature_map_count, entry_count, backward_packed_config_count);

#define launch_backprop_kernel_const_const(dimension_count_const, window_width, block_size_const) \
	switch (window_width) \
		{ \
		case 1: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 1, block_size_const); \
			break; \
		case 2: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 2, block_size_const); \
			break; \
		case 3: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 3, block_size_const); \
			break; \
		case 4: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 4, block_size_const); \
			break; \
		case 5: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 5, block_size_const); \
			break; \
		case 6: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 6, block_size_const); \
			break; \
		case 7: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 7, block_size_const); \
			break; \
		case 8: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 8, block_size_const); \
			break; \
		case 9: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 9, block_size_const); \
			break; \
		case 10: \
			launch_backprop_exact_kernel_const_const_const(dimension_count_const, 10, block_size_const); \
			break; \
		default: \
			launch_backprop_generic_kernel_const_const(dimension_count_const, block_size_const); \
			break; \
		};

#define launch_backprop_kernel(dimension_count_const, window_width, block_size) \
	switch (block_size) \
		{ \
		case 1: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, 1); \
			break; \
		case 2: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, 2); \
			break; \
		case 3: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, 3); \
			break; \
		case 4: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, 4); \
			break; \
		case 5: \
			launch_backprop_kernel_const_const(dimension_count_const, window_width, 5); \
			break; \
		};

#define launch_update_exact_kernel_const_const(dimension_count_const, window_width_const) \
	convolution_update_weights_exact_hess_kernel_kepler<dimension_count_const, window_width_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*hessian_data[0], input_squared_tex, output_tex, packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific_striped.feature_map_count, input_elem_count_per_entry_striped, output_elem_count_per_entry_striped, entry_count, block_size, updater_packed_config_count);

#define launch_update_generic_kernel_const(dimension_count_const) \
	convolution_update_weights_generic_hess_kernel_kepler<dimension_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*hessian_data[0], input_squared_tex, output_tex, packed_config_list, output_sizes, input_sizes, window_sizes, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_configuration_specific_striped.feature_map_count, output_configuration_specific_striped.feature_map_count, input_elem_count_per_entry_striped, output_elem_count_per_entry_striped, entry_count, block_size, updater_packed_config_count);

#define launch_update_kernel(dimension_count_const, window_width) \
	switch (window_width) \
		{ \
		case 1: \
			launch_update_exact_kernel_const_const(dimension_count_const, 1); \
			break; \
		case 2: \
			launch_update_exact_kernel_const_const(dimension_count_const, 2); \
			break; \
		case 3: \
			launch_update_exact_kernel_const_const(dimension_count_const, 3); \
			break; \
		case 4: \
			launch_update_exact_kernel_const_const(dimension_count_const, 4); \
			break; \
		case 5: \
			launch_update_exact_kernel_const_const(dimension_count_const, 5); \
			break; \
		case 6: \
			launch_update_exact_kernel_const_const(dimension_count_const, 6); \
			break; \
		case 7: \
			launch_update_exact_kernel_const_const(dimension_count_const, 7); \
			break; \
		case 8: \
			launch_update_exact_kernel_const_const(dimension_count_const, 8); \
			break; \
		case 9: \
			launch_update_exact_kernel_const_const(dimension_count_const, 9); \
			break; \
		case 10: \
			launch_update_exact_kernel_const_const(dimension_count_const, 10); \
			break; \
		default: \
			launch_update_generic_kernel_const(dimension_count_const); \
			break; \
		};

		template<int dimension_count>
		class convolution_layer_hessian_cuda_kepler : public layer_hessian_cuda
		{
		public:
			convolution_layer_hessian_cuda_kepler()
			{
			}

			virtual ~convolution_layer_hessian_cuda_kepler()
			{
			}

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count)
			{
				cuda_util::copy_to_striped(
					*cuda_config,
					*input_neurons_buffer,
					*additional_buffers[0],
					input_elem_count_per_feature_map,
					input_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				cuda_texture weights_tex(data[0], 2);
				cuda_texture input_tex(additional_buffers[0], 2);

				const packed_config<dimension_count> * packed_config_list = static_cast<const packed_config<dimension_count> *>((const void *)*additional_buffers[2]);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_x_block_count,
					forward_packed_config_count,
					entry_count);

				launch_kernel(dimension_count, window_sizes[0], forward_x_block_size);
			}

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_squared,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count)
			{
				cuda_texture weights_squared_tex(data_squared[0], 2);
				cuda_texture output_tex(additional_buffers[1], 2);

				const packed_config<dimension_count> * packed_config_list = static_cast<const packed_config<dimension_count> *>((const void *)*additional_buffers[4]);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					backward_x_block_count,
					backward_packed_config_count,
					entry_count);

				launch_backprop_kernel(dimension_count, window_sizes[0], backward_x_block_size);
			}

			virtual void enqueue_update_hessian(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& hessian_data,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count)
			{
				cuda_util::copy_to_striped(
					*cuda_config,
					*output_errors_buffer,
					*additional_buffers[1],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				// Update weights
				{
					// Store input neurons multiplied element-wise by themselves
					cuda_util::multiply_by_itself(
						*cuda_config,
						*additional_buffers[0],
						*additional_buffers[0],
						input_elem_count_per_entry_striped * 2 * entry_count,
						stream_id);

					cuda_texture input_squared_tex(additional_buffers[0], 2);
					cuda_texture output_tex(additional_buffers[1], 2);

					int block_size = get_weights_update_block_size(entry_count);
					int block_count = (entry_count + block_size - 1) / block_size;

					const packed_config<updater_dimension_count> * packed_config_list = static_cast<const packed_config<updater_dimension_count> *>((const void *)*additional_buffers[3]);

					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						updater_packed_config_count,
						block_count,
						1);

					launch_update_kernel(dimension_count, window_sizes[0]);
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
					convolution_update_biases_hess_kernel_kepler<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*hessian_data[1],
						*output_errors_buffer,
						block_size,
						output_elem_count_per_feature_map,
						output_configuration_specific.feature_map_count,
						entry_count);
				}
			}

			virtual std::vector<const_cuda_linear_buffer_device_smart_ptr> get_data(const_layer_data_smart_ptr host_data) const
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

				if (host_data->size() != 2)
					return res;

				unsigned int window_total_size = 1;
				for(int i = 0; i < dimension_count; ++i)
					window_total_size *= window_sizes[i];
				unsigned int weight_count = output_configuration_specific.feature_map_count * input_configuration_specific.feature_map_count * window_total_size;
				if (host_data->at(0).size() != weight_count)
					return res;

				if (host_data->at(1).size() != output_configuration_specific.feature_map_count)
					return res;

				unsigned int input_feature_map_count_striped = input_configuration_specific_striped.feature_map_count;
				unsigned int weight_count_striped = output_configuration_specific.feature_map_count * input_feature_map_count_striped * 2 * window_total_size;

				std::vector<float> weights_striped(weight_count_striped, 0.0F);
				const std::vector<float>& src = host_data->at(0);
				unsigned int src_offset = 0;
				unsigned int dst_offset = 0;
				for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_configuration_specific.feature_map_count; ++output_feature_map_id)
				{
					for(unsigned int input_feature_map_id_striped = 0; input_feature_map_id_striped < input_feature_map_count_striped; ++input_feature_map_id_striped, dst_offset += window_total_size * 2)
					{
						bool second_feature_map_present = (input_feature_map_id_striped * 2 + 1 < input_configuration_specific.feature_map_count);
						for(int dst_elem_id = 0; dst_elem_id < window_total_size; ++dst_elem_id)
						{
							weights_striped[dst_offset + dst_elem_id * 2] = src[src_offset + dst_elem_id];
							if (second_feature_map_present)
								weights_striped[dst_offset + dst_elem_id * 2 + 1] = src[src_offset + dst_elem_id + window_total_size];
						}

						src_offset += window_total_size * (second_feature_map_present ? 2 : 1);
					}
				}
				{
					size_t buffer_size = weights_striped.size() * sizeof(float);
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*weights_striped.begin()), buffer_size, cudaMemcpyHostToDevice));
					res.push_back(new_buf);
				}

				{
					size_t buffer_size = host_data->at(1).size() * sizeof(float);
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*host_data->at(1).begin()), buffer_size, cudaMemcpyHostToDevice));
					res.push_back(new_buf);
				}

				return res;
			}

			virtual std::vector<const_cuda_linear_buffer_device_smart_ptr> get_data_squared(const_layer_data_smart_ptr host_data) const
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

				if (!backprop_required)
					return res;

				if (host_data->size() != 2)
					return res;

				unsigned int window_total_size = 1;
				for(int i = 0; i < dimension_count; ++i)
					window_total_size *= window_sizes[i];
				unsigned int weight_count = output_configuration_specific.feature_map_count * input_configuration_specific.feature_map_count * window_total_size;
				if (host_data->at(0).size() != weight_count)
					return res;

				if (host_data->at(1).size() != output_configuration_specific.feature_map_count)
					return res;

				unsigned int output_feature_map_count_striped = output_configuration_specific_striped.feature_map_count;
				unsigned int weight_count_striped = input_configuration_specific.feature_map_count * output_feature_map_count_striped * 2 * window_total_size;

				std::vector<float> weights_striped(weight_count_striped, 0.0F);
				const std::vector<float>& src = host_data->at(0);
				unsigned int dst_offset = 0;
				unsigned int weight_count_per_output_feature_map = window_total_size * input_configuration_specific.feature_map_count;
				for(unsigned int input_feature_map_id = 0; input_feature_map_id < input_configuration_specific.feature_map_count; ++input_feature_map_id)
				{
					for(unsigned int output_feature_map_id_striped = 0; output_feature_map_id_striped < output_feature_map_count_striped; ++output_feature_map_id_striped, dst_offset += window_total_size * 2)
					{
						unsigned int src_offset = (output_feature_map_id_striped * 2 * input_configuration_specific.feature_map_count + input_feature_map_id) * window_total_size;

						bool second_feature_map_present = (output_feature_map_id_striped * 2 + 1 < output_configuration_specific.feature_map_count);
						for(int dst_elem_id = 0; dst_elem_id < window_total_size; ++dst_elem_id)
						{
							weights_striped[dst_offset + dst_elem_id * 2] = src[src_offset + dst_elem_id];
							if (second_feature_map_present)
								weights_striped[dst_offset + dst_elem_id * 2 + 1] = src[src_offset + dst_elem_id + weight_count_per_output_feature_map];
						}
					}
				}
				{
					size_t buffer_size = weights_striped.size() * sizeof(float);
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*weights_striped.begin()), buffer_size, cudaMemcpyHostToDevice));
					cuda_util::multiply_by_itself(
						*cuda_config,
						*new_buf,
						*new_buf,
						new_buf->get_size() / sizeof(float),
						0);
					res.push_back(new_buf);
				}

				{
					size_t buffer_size = host_data->at(1).size() * sizeof(float);
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*host_data->at(1).begin()), buffer_size, cudaMemcpyHostToDevice));
					cuda_util::multiply_by_itself(
						*cuda_config,
						*new_buf,
						*new_buf,
						new_buf->get_size() / sizeof(float),
						0);
					res.push_back(new_buf);
				}
				cuda_safe_call(cudaStreamSynchronize(0));

				return res;
			}

		protected:
			static const int updater_dimension_count = (dimension_count * 2 + 2);

			virtual void hessian_configured()
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

					forward_packed_config_count = forward_output_feature_map_block_count;
					for(int i = 1; i < dimension_count; ++i)
						forward_packed_config_count *= output_sizes[i];
				}

				{
					output_configuration_specific_striped = cuda_util::get_layer_configuration_specific_striped(output_configuration_specific);
					output_elem_count_per_entry_striped = output_configuration_specific_striped.get_neuron_count();

					backward_x_block_size = get_block_size(input_configuration_specific.dimension_sizes[0]);
					backward_x_block_count = (input_configuration_specific.dimension_sizes[0] + backward_x_block_size - 1) / backward_x_block_size;
					backward_input_feature_map_block_count = (input_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

					backward_packed_config_count = backward_input_feature_map_block_count;
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
				}
			}

			virtual bool is_in_place_backprop() const
			{
				return false;
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const
			{
				std::vector<size_t> res;

				res.push_back(input_elem_count_per_entry_striped * sizeof(float2));
				res.push_back(output_elem_count_per_entry_striped * sizeof(float2));

				return res;
			}

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const
			{
				std::vector<unsigned int> res;

				res.push_back(input_elem_count_per_entry_striped);
				res.push_back(output_elem_count_per_entry_striped);

				return res;
			}

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const
			{
				std::vector<size_t> res;

				res.push_back(sizeof(packed_config<dimension_count>) * forward_packed_config_count);

				res.push_back(sizeof(packed_config<updater_dimension_count>) * updater_packed_config_count);

				if (backprop_required)
				{
					res.push_back(sizeof(packed_config<dimension_count>) * backward_packed_config_count);
				}

				return res;
			}

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
			{
				{
					std::vector<packed_config<dimension_count> > task_list;
					if (dimension_count > 1)
					{
						nnforge_array<int, dimension_count - 1> size_list;
						for(int i = 0; i < dimension_count - 1; ++i)
							size_list[i] = output_sizes[i + 1];
						std::vector<nnforge_array<int, dimension_count - 1> > ordered_list;
						sequential_curve<dimension_count - 1>::fill_pattern(size_list, ordered_list);
						packed_config<dimension_count> new_elem;
						for(int output_feature_map_block_id = 0; output_feature_map_block_id < forward_output_feature_map_block_count; ++output_feature_map_block_id)
						{
							new_elem.set_val(dimension_count - 1, output_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
							for(int j = 0; j < ordered_list.size(); ++j)
							{
								const nnforge_array<int, dimension_count - 1>& spatial_dimensions = ordered_list[j];
								for(int i = 0; i < dimension_count - 1; ++i)
									new_elem.set_val(i, spatial_dimensions[i]);
								task_list.push_back(new_elem);
							}
						}
					}
					else
					{
						packed_config<dimension_count> new_elem;
						for(int output_feature_map_block_id = 0; output_feature_map_block_id < forward_output_feature_map_block_count; ++output_feature_map_block_id)
						{
							new_elem.set_val(dimension_count - 1, output_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
							task_list.push_back(new_elem);
						}
					}
					cuda_safe_call(cudaMemcpy(*additional_buffers[2], &(*task_list.begin()), sizeof(packed_config<dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
				}

				{
					std::vector<packed_config<updater_dimension_count> > task_list;

					nnforge_array<int, dimension_count * 2> size_list;
					for(int i = 1; i < dimension_count; ++i)
					{
						size_list[i - 1] = window_sizes[i];
						size_list[(dimension_count - 1) + i - 1] = output_sizes[i];
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
					std::vector<packed_config<dimension_count> > task_list;
					if (dimension_count > 1)
					{
						nnforge_array<int, dimension_count - 1> size_list;
						for(int i = 0; i < dimension_count - 1; ++i)
							size_list[i] = input_sizes[i + 1];
						std::vector<nnforge_array<int, dimension_count - 1> > ordered_list;
						sequential_curve<dimension_count - 1>::fill_pattern(size_list, ordered_list);
						packed_config<dimension_count> new_elem;
						for(int input_feature_map_block_id = 0; input_feature_map_block_id < backward_input_feature_map_block_count; ++input_feature_map_block_id)
						{
							new_elem.set_val(dimension_count - 1, input_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
							for(int j = 0; j < ordered_list.size(); ++j)
							{
								const nnforge_array<int, dimension_count - 1>& spatial_dimensions = ordered_list[j];
								for(int i = 0; i < dimension_count - 1; ++i)
									new_elem.set_val(i, spatial_dimensions[i]);
								task_list.push_back(new_elem);
							}
						}
					}
					else
					{
						packed_config<dimension_count> new_elem;
						for(int input_feature_map_block_id = 0; input_feature_map_block_id < backward_input_feature_map_block_count; ++input_feature_map_block_id)
						{
							new_elem.set_val(dimension_count - 1, input_feature_map_block_id * FEATURE_MAP_BLOCK_SIZE);
							task_list.push_back(new_elem);
						}
					}
					cuda_safe_call(cudaMemcpy(*additional_buffers[4], &(*task_list.begin()), sizeof(packed_config<dimension_count>) * task_list.size(), cudaMemcpyHostToDevice));
				}
			}

		private:
			int get_block_size(int width)
			{
				int block_count = (width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
				int block_size = (width + block_count - 1) / block_count;
				return block_size;
			}

			int get_bias_update_block_size(int entry_count)
			{
				int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
				return block_size;
			}

			int get_weights_update_block_size(int entry_count)
			{
				int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count)) * 2.0F), 1), entry_count);
				int block_count = (entry_count + block_size - 1) / block_size;
				block_size = (entry_count + block_count - 1) / block_count;
				return block_size;
			}

		private:
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> window_sizes;

			layer_configuration_specific input_configuration_specific_striped;
			layer_configuration_specific output_configuration_specific_striped;
			unsigned int input_elem_count_per_entry_striped;
			unsigned int output_elem_count_per_entry_striped;

			int forward_x_block_size;
			int forward_x_block_count;
			int forward_output_feature_map_block_count;
			int forward_packed_config_count;

			int backward_x_block_size;
			int backward_x_block_count;
			int backward_input_feature_map_block_count;
			int backward_packed_config_count;

			int updater_packed_config_count;
			int updater_window_x_block_count;
		};
	}
}
