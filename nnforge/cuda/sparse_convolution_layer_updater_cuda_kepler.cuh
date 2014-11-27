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
#include "cuda_texture.h"
#include "neural_network_cuda_exception.h"

#include "../sparse_convolution_layer.h"
#include "../nn_types.h"

#define MAX_DIMENSION_COUNT 4

#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4

#define MAX_EXACT_GRADIENT_UPDATE_WINDOW_WIDTH_HEIGHT 7

namespace nnforge
{
	namespace cuda
	{
		struct __align__(8) row_index_weight_block_id_pair
		{
			int row_index;
			int weight_block_id;
		};

		struct __align__(8) row_index_col_index_pair
		{
			int row_index;
			int col_index;
		};

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int WINDOW_HEIGHT>
		__launch_bounds__(256, 4)
		__global__ void sparse_convolution_tex_exact_blocked_upd_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			const float * __restrict biases,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_elem_count_per_feature_map,
			int entry_count,
			int input_elem_offset,
			int block_count_per_feature_map,
			int weight_count_per_block,
			unsigned int dummy)
		{
			int neuron_output_feature_map_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = neuron_output_feature_map_pair_id / block_count_per_feature_map;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (output_feature_map_id < output_feature_map_count);
			if (in_bounds)
			{
				int neuron_id = neuron_output_feature_map_pair_id - block_count_per_feature_map * output_feature_map_id;

				int xyzw_output[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / output_block_sizes[i];
					xyzw_output[i] = remainder - output_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_output[DIMENSION_COUNT - 1] = remainder;
				xyzw_output[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_output[1] *= BLOCK_HEIGHT;

				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw[i] = xyzw_output[i] - left_zero_padding[i];
				}

				int start_column_index = __load_nc(row_ptrs + output_feature_map_id);
				int end_column_index = __load_nc(row_ptrs + output_feature_map_id + 1);

				const float * current_weights = weights + weight_count_per_block * start_column_index;

				int input_elem_id_base = entry_id * input_feature_map_count;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id_base = input_elem_id_base * input_sizes[i] + xyzw[i];
				input_elem_id_base += input_elem_offset;

				float sums[(DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1)][BLOCK_WIDTH];

				float bias = biases[output_feature_map_id];
				#pragma unroll
				for(int i = 0; i < (DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1); ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_WIDTH; ++j)
						sums[i][j] = bias;

				unsigned int valid_positions[(((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1) * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(valid_positions) / sizeof(unsigned int); ++i)
					valid_positions[i] = 0;
				#pragma unroll
				for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++input_yy)
				{
					int input_y = input_yy + ((DIMENSION_COUNT > 1) ? xyzw[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)input_y < (unsigned int)input_sizes[1]) : true;
					#pragma unroll
					for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
					{
						int input_x = input_xx + xyzw[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)input_x < (unsigned int)input_sizes[0]));
						if (b_fit0)
						{
							int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
							valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				for(int nnz_index = start_column_index; nnz_index < end_column_index; ++nnz_index)
				{
					#pragma unroll
					for(int i = 0; i < sizeof(valid_positions) / sizeof(unsigned int); ++i)
						valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					int input_feature_map_id = column_indices[nnz_index];
					int input_elem_id = input_elem_id_base + input_feature_map_id * input_elem_count_per_feature_map;

					for(int input_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); input_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)input_w < (unsigned int)input_sizes[3]) : true;
						for(int input_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); input_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)input_z < (unsigned int)input_sizes[2])) : true;

							float input_local_buf[(DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1][WINDOW_WIDTH + BLOCK_WIDTH - 1];

							#pragma unroll
							for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++input_yy)
							{
								#pragma unroll
								for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
								{
									int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
									bool b_fit0 = b_fit2 && ((valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
									int current_offset = input_elem_id + input_yy * input_sizes[0] + input_xx;
									input_local_buf[input_yy][input_xx] = tex1Dfetch<float>(input_tex, b_fit0 ? current_offset : -1);
								}
							}

							#pragma unroll
							for(int input_yy = 0; input_yy < WINDOW_HEIGHT; ++input_yy)
							{
								#pragma unroll
								for(int input_xx = 0; input_xx < WINDOW_WIDTH; ++input_xx)
								{
									float weight = __load_nc(current_weights);

									#pragma unroll
									for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
									{
										#pragma unroll
										for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
										{
											float input_val = input_local_buf[input_yy + pos_y][input_xx + pos_x];
											sums[pos_y][pos_x] += weight * input_val;
										}
									}

									++current_weights;
								}
							}

							if (DIMENSION_COUNT > 2)
								input_elem_id += input_sizes[0] * input_sizes[1];
						} // for input_z
						if (DIMENSION_COUNT > 3)
							input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - window_sizes[2]);
					} // for input_w
				} // for nnz_index

				int output_offset = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_offset = output_offset * output_sizes[i] + xyzw_output[i];

				#pragma unroll
				for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
				{
					if ((DIMENSION_COUNT > 1) ? (pos_y < output_sizes[1] - xyzw_output[1]) : true)
					{
						#pragma unroll
						for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
						{
							if (pos_x < output_sizes[0] - xyzw_output[0])
							{
								output[output_offset + pos_x] = sums[pos_y][pos_x];
							}
						}
					}
					output_offset += output_sizes[0];
				}
			} // if (in_bounds)
		}

		template<int DIMENSION_COUNT>
		__launch_bounds__(256, 4)
		__global__ void sparse_convolution_tex_generic_blocked_upd_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			const float * __restrict biases,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_elem_count_per_feature_map,
			int entry_count,
			int input_elem_offset,
			int block_count_per_feature_map,
			int weight_count_per_block)
		{
			int neuron_output_feature_map_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = neuron_output_feature_map_pair_id / block_count_per_feature_map;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (output_feature_map_id < output_feature_map_count);
			if (in_bounds)
			{
				int neuron_id = neuron_output_feature_map_pair_id - block_count_per_feature_map * output_feature_map_id;

				int xyzw_output[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / output_block_sizes[i];
					xyzw_output[i] = remainder - output_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_output[DIMENSION_COUNT - 1] = remainder;
				xyzw_output[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_output[1] *= BLOCK_HEIGHT;

				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw[i] = xyzw_output[i] - left_zero_padding[i];
				}

				int start_column_index = __load_nc(row_ptrs + output_feature_map_id);
				int end_column_index = __load_nc(row_ptrs + output_feature_map_id + 1);

				const float * current_weights = weights + weight_count_per_block * start_column_index;

				int input_elem_id_base = entry_id * input_feature_map_count;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id_base = input_elem_id_base * input_sizes[i] + xyzw[i];
				input_elem_id_base += input_elem_offset;

				float sums[(DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1)][BLOCK_WIDTH];

				float bias = biases[output_feature_map_id];
				#pragma unroll
				for(int i = 0; i < (DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1); ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_WIDTH; ++j)
						sums[i][j] = bias;

				for(int nnz_index = start_column_index; nnz_index < end_column_index; ++nnz_index)
				{
					int input_feature_map_id = column_indices[nnz_index];
					int input_elem_id = input_elem_id_base + input_feature_map_id * input_elem_count_per_feature_map;

					for(int input_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); input_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)input_w < (unsigned int)input_sizes[3]) : true;
						for(int input_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); input_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)input_z < (unsigned int)input_sizes[2])) : true;

							#pragma unroll 2
							for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? window_sizes[1] : 1); ++input_yy)
							{
								#pragma unroll 2
								for(int input_xx = 0; input_xx < window_sizes[0]; ++input_xx)
								{
									float weight = __load_nc(current_weights);

									#pragma unroll
									for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
									{
										bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && ((unsigned int)(xyzw[1] + input_yy + pos_y) < (unsigned int)input_sizes[1])) : true;
										#pragma unroll
										for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
										{
											bool b_fit0 = (b_fit1 && ((unsigned int)(xyzw[0] + input_xx + pos_x) < (unsigned int)input_sizes[0]));
											int current_offset = input_elem_id + (input_yy + pos_y) * input_sizes[0] + input_xx + pos_x;
											float input_val = tex1Dfetch<float>(input_tex, b_fit0 ? current_offset : -1);
											sums[pos_y][pos_x] += weight * input_val;
										}
									}

									++current_weights;
								}
							}

							if (DIMENSION_COUNT > 2)
								input_elem_id += input_sizes[0] * input_sizes[1];
						} // for input_z
						if (DIMENSION_COUNT > 3)
							input_elem_id += input_sizes[1] * input_sizes[0] * (input_sizes[2] - window_sizes[2]);
					} // for input_w
				} // for nnz_index

				int output_offset = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_offset = output_offset * output_sizes[i] + xyzw_output[i];

				#pragma unroll
				for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
				{
					if ((DIMENSION_COUNT > 1) ? (pos_y < output_sizes[1] - xyzw_output[1]) : true)
					{
						#pragma unroll
						for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
						{
							if (pos_x < output_sizes[0] - xyzw_output[0])
							{
								output[output_offset + pos_x] = sums[pos_y][pos_x];
							}
						}
					}
					output_offset += output_sizes[0];
				}
			} // if (in_bounds)
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int WINDOW_HEIGHT>
		__launch_bounds__(256, 4)
		__global__ void sparse_convolution_backprop_tex_exact_blocked_upd_kernel_kepler(
			float * __restrict input,
			cudaTextureObject_t output_tex,
			const float * __restrict weights,
			const row_index_weight_block_id_pair * __restrict row_index_weight_block_id_pairs,
			const int * __restrict col_ptrs,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> input_block_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_elem_count_per_feature_map,
			int entry_count,
			int block_count_per_feature_map,
			int weight_count_per_block,
			unsigned int dummy)
		{
			int neuron_input_feature_map_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
			int input_feature_map_id = neuron_input_feature_map_pair_id / block_count_per_feature_map;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (input_feature_map_id < input_feature_map_count);
			if (in_bounds)
			{
				int neuron_id = neuron_input_feature_map_pair_id - block_count_per_feature_map * input_feature_map_id;

				int xyzw_input[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / input_block_sizes[i];
					xyzw_input[i] = remainder - input_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_input[DIMENSION_COUNT - 1] = remainder;
				xyzw_input[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_input[1] *= BLOCK_HEIGHT;

				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw[i] = xyzw_input[i] + left_zero_padding[i] - (window_sizes[i] - 1);
				}

				int start_row_index = __load_nc(col_ptrs + input_feature_map_id);
				int end_row_index = __load_nc(col_ptrs + input_feature_map_id + 1);

				int output_elem_id_base = entry_id * output_feature_map_count;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id_base = output_elem_id_base * output_sizes[i] + xyzw[i];

				float sums[(DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1)][BLOCK_WIDTH];
				for(int i = 0; i < (DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1); ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_WIDTH; ++j)
						sums[i][j] = 0.0F;

				unsigned int valid_positions[(((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1) * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(valid_positions) / sizeof(unsigned int); ++i)
					valid_positions[i] = 0;
				#pragma unroll
				for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++output_yy)
				{
					int output_y = output_yy + ((DIMENSION_COUNT > 1) ? xyzw[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)output_y < (unsigned int)output_sizes[1]) : true;
					#pragma unroll
					for(int output_xx = 0; output_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++output_xx)
					{
						int output_x = output_xx + xyzw[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)output_x < (unsigned int)output_sizes[0]));
						if (b_fit0)
						{
							int pos_total = output_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + output_xx;
							valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				for(int nnz_index = start_row_index; nnz_index < end_row_index; ++nnz_index)
				{
					#pragma unroll
					for(int i = 0; i < sizeof(valid_positions) / sizeof(unsigned int); ++i)
						valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					row_index_weight_block_id_pair rw_pair = row_index_weight_block_id_pairs[nnz_index];
					int output_feature_map_id = rw_pair.row_index;
					int weight_block_id = rw_pair.weight_block_id;
					const float * current_weights = weights + (weight_count_per_block * (weight_block_id + 1) - 1);

					int output_elem_id = output_elem_id_base + output_feature_map_id * output_elem_count_per_feature_map;

					for(int output_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); output_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++output_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)output_w < (unsigned int)output_sizes[3]) : true;
						for(int output_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); output_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++output_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)output_z < (unsigned int)output_sizes[2])) : true;

							float output_local_buf[(DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1][WINDOW_WIDTH + BLOCK_WIDTH - 1];

							#pragma unroll
							for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++output_yy)
							{
								#pragma unroll
								for(int output_xx = 0; output_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++output_xx)
								{
									int pos_total = output_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + output_xx;
									bool b_fit0 = b_fit2 && ((valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
									int current_offset = output_elem_id + output_yy * output_sizes[0] + output_xx;
									output_local_buf[output_yy][output_xx] = tex1Dfetch<float>(output_tex, b_fit0 ? current_offset : -1);
								}
							}

							#pragma unroll
							for(int output_yy = 0; output_yy < WINDOW_HEIGHT; ++output_yy)
							{
								#pragma unroll
								for(int output_xx = 0; output_xx < WINDOW_WIDTH; ++output_xx)
								{
									float weight = __load_nc(current_weights);

									#pragma unroll
									for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
									{
										#pragma unroll
										for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
										{
											float output_val = output_local_buf[output_yy + pos_y][output_xx + pos_x];
											sums[pos_y][pos_x] += weight * output_val;
										}
									}

									--current_weights;
								}
							}

							if (DIMENSION_COUNT > 2)
								output_elem_id += output_sizes[0] * output_sizes[1];
						} // for output_z
						if (DIMENSION_COUNT > 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (output_sizes[2] - window_sizes[2]);
					} // for output_w
				} // for nnz_index

				int input_offset = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_offset = input_offset * input_sizes[i] + xyzw_input[i];

				#pragma unroll
				for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
				{
					if ((DIMENSION_COUNT > 1) ? (pos_y < input_sizes[1] - xyzw_input[1]) : true)
					{
						#pragma unroll
						for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
						{
							if (pos_x < input_sizes[0] - xyzw_input[0])
							{
								input[input_offset + pos_x] = sums[pos_y][pos_x];
							}
						}
					}
					input_offset += input_sizes[0];
				}
			} // if (in_bounds)
		}

		template<int DIMENSION_COUNT>
		__launch_bounds__(256, 4)
		__global__ void sparse_convolution_backprop_tex_generic_blocked_upd_kernel_kepler(
			float * __restrict input,
			cudaTextureObject_t output_tex,
			const float * __restrict weights,
			const row_index_weight_block_id_pair * __restrict row_index_weight_block_id_pairs,
			const int * __restrict col_ptrs,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> input_block_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_elem_count_per_feature_map,
			int entry_count,
			int block_count_per_feature_map,
			int weight_count_per_block)
		{
			int neuron_input_feature_map_pair_id = blockIdx.x * blockDim.x + threadIdx.x;
			int input_feature_map_id = neuron_input_feature_map_pair_id / block_count_per_feature_map;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			bool in_bounds = (entry_id < entry_count) && (input_feature_map_id < input_feature_map_count);
			if (in_bounds)
			{
				int neuron_id = neuron_input_feature_map_pair_id - block_count_per_feature_map * input_feature_map_id;

				int xyzw_input[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / input_block_sizes[i];
					xyzw_input[i] = remainder - input_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_input[DIMENSION_COUNT - 1] = remainder;
				xyzw_input[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_input[1] *= BLOCK_HEIGHT;

				int xyzw[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw[i] = xyzw_input[i] + left_zero_padding[i] - (window_sizes[i] - 1);
				}

				int start_row_index = __load_nc(col_ptrs + input_feature_map_id);
				int end_row_index = __load_nc(col_ptrs + input_feature_map_id + 1);

				int output_elem_id_base = entry_id * output_feature_map_count;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id_base = output_elem_id_base * output_sizes[i] + xyzw[i];

				float sums[(DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1)][BLOCK_WIDTH];
				for(int i = 0; i < (DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1); ++i)
					#pragma unroll
					for(int j = 0; j < BLOCK_WIDTH; ++j)
						sums[i][j] = 0.0F;

				for(int nnz_index = start_row_index; nnz_index < end_row_index; ++nnz_index)
				{
					row_index_weight_block_id_pair rw_pair = row_index_weight_block_id_pairs[nnz_index];
					int output_feature_map_id = rw_pair.row_index;
					int weight_block_id = rw_pair.weight_block_id;
					const float * current_weights = weights + (weight_count_per_block * (weight_block_id + 1) - 1);

					int output_elem_id = output_elem_id_base + output_feature_map_id * output_elem_count_per_feature_map;

					for(int output_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); output_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++output_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)output_w < (unsigned int)output_sizes[3]) : true;
						for(int output_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); output_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++output_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)output_z < (unsigned int)output_sizes[2])) : true;

							#pragma unroll 2
							for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? window_sizes[1] : 1); ++output_yy)
							{
								#pragma unroll 2
								for(int output_xx = 0; output_xx < window_sizes[0]; ++output_xx)
								{
									float weight = __load_nc(current_weights);

									#pragma unroll
									for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
									{
										bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && ((unsigned int)(xyzw[1] + output_yy + pos_y) < (unsigned int)output_sizes[1])) : true;
										#pragma unroll
										for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
										{
											bool b_fit0 = (b_fit1 && ((unsigned int)(xyzw[0] + output_xx + pos_x) < (unsigned int)output_sizes[0]));
											int current_offset = output_elem_id + (output_yy + pos_y) * output_sizes[0] + output_xx + pos_x;
											float output_val = tex1Dfetch<float>(output_tex, b_fit0 ? current_offset : -1);
											sums[pos_y][pos_x] += weight * output_val;
										}
									}

									--current_weights;
								}
							}

							if (DIMENSION_COUNT > 2)
								output_elem_id += output_sizes[0] * output_sizes[1];
						} // for output_z
						if (DIMENSION_COUNT > 3)
							output_elem_id += output_sizes[1] * output_sizes[0] * (output_sizes[2] - window_sizes[2]);
					} // for output_w
				} // for nnz_index

				int input_offset = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_offset = input_offset * input_sizes[i] + xyzw_input[i];

				#pragma unroll
				for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
				{
					if ((DIMENSION_COUNT > 1) ? (pos_y < input_sizes[1] - xyzw_input[1]) : true)
					{
						#pragma unroll
						for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
						{
							if (pos_x < input_sizes[0] - xyzw_input[0])
							{
								input[input_offset + pos_x] = sums[pos_y][pos_x];
							}
						}
					}
					input_offset += input_sizes[0];
				}
			} // if (in_bounds)
		}

		extern __shared__ float arr[];
		__global__ void sparse_convolution_update_biases_upd_kernel_kepler(
			float * __restrict gradient_biases,
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
				atomicAdd(gradient_biases + output_feature_map_id, arr[0]);
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int WINDOW_HEIGHT>
		__launch_bounds__(256, 3)
		__global__ void sparse_convolution_update_gradient_tex_exact_blocked_upd_kernel_kepler(
			float * __restrict gradients,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t input_tex,
			const row_index_col_index_pair * __restrict row_index_col_index_pairs,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int entry_group_size,
			int input_elem_offset,
			int block_count_per_output_feature_map,
			int feature_map_pair_count,
			int input_elem_count_per_entry,
			int output_elem_count_per_entry,
			unsigned int dummy)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

			float weight[DIMENSION_COUNT];
			weight[0] = 0;
			if (DIMENSION_COUNT > 1)
				weight[1] = 0;
			int weight_zw_feature_map_pair_id = blockIdx.y * blockDim.y + threadIdx.y;
			int feature_map_pair_id;
			bool weight_zw_feature_map_pair_fit;
			if (DIMENSION_COUNT > 2)
			{
				int remainder = weight_zw_feature_map_pair_id / feature_map_pair_count;
				feature_map_pair_id = weight_zw_feature_map_pair_id - feature_map_pair_count * remainder;

				for(int i = 2; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / window_sizes[i];
					weight[i] = remainder - window_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				weight[DIMENSION_COUNT - 1] = remainder;
				weight_zw_feature_map_pair_fit = (weight[DIMENSION_COUNT - 1] < window_sizes[DIMENSION_COUNT - 1]);
			}
			else
			{
				feature_map_pair_id = weight_zw_feature_map_pair_id;
				weight_zw_feature_map_pair_fit = (feature_map_pair_id < feature_map_pair_count);
			}

			int entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * entry_group_size;

			bool in_bounds = weight_zw_feature_map_pair_fit && (neuron_id < block_count_per_output_feature_map) && (entry_id < entry_count);
			if (in_bounds)
			{
				int xyzw_output[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / output_block_sizes[i];
					xyzw_output[i] = remainder - output_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_output[DIMENSION_COUNT - 1] = remainder;
				xyzw_output[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_output[1] *= BLOCK_HEIGHT;

				int xyzw_input[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw_input[i] = xyzw_output[i] - left_zero_padding[i] + weight[i];
				}

				#pragma unroll
				for(int i = 2; i < DIMENSION_COUNT; ++i)
				{
					if ((unsigned int)xyzw_input[i] >= (unsigned int)input_sizes[i]);
						return;
				}

				row_index_col_index_pair rc = row_index_col_index_pairs[feature_map_pair_id];
				int output_feature_map_id = rc.row_index;
				int input_feature_map_id = rc.col_index;

				int input_elem_id = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw_input[i];
				input_elem_id += input_elem_offset;

				int output_elem_id = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw_output[i];

				float local_gradients[WINDOW_HEIGHT][WINDOW_WIDTH];
				#pragma unroll
				for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
				{
					#pragma unroll
					for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
					{
						local_gradients[weight_y][weight_x] = 0.0F;
					}
				}

				unsigned int input_valid_positions[(((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1) * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(input_valid_positions) / sizeof(unsigned int); ++i)
					input_valid_positions[i] = 0;
				#pragma unroll
				for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++input_yy)
				{
					int input_y = input_yy + ((DIMENSION_COUNT > 1) ? xyzw_input[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)input_y < (unsigned int)input_sizes[1]) : true;
					#pragma unroll
					for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
					{
						int input_x = input_xx + xyzw_input[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)input_x < (unsigned int)input_sizes[0]));
						if (b_fit0)
						{
							int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
							input_valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				unsigned int output_valid_positions[(((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1) * BLOCK_WIDTH + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(output_valid_positions) / sizeof(unsigned int); ++i)
					output_valid_positions[i] = 0;
				#pragma unroll
				for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
				{
					int output_y = output_yy + ((DIMENSION_COUNT > 1) ? xyzw_output[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)output_y < (unsigned int)output_sizes[1]) : true;
					#pragma unroll
					for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
					{
						int output_x = output_xx + xyzw_output[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)output_x < (unsigned int)output_sizes[0]));
						if (b_fit0)
						{
							int pos_total = output_yy * BLOCK_WIDTH + output_xx;
							output_valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				int it_count = min(entry_group_size, entry_count - entry_id);
				for(int it = 0; it < it_count; ++it)
				{
					#pragma unroll
					for(int i = 0; i < sizeof(input_valid_positions) / sizeof(unsigned int); ++i)
						input_valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					#pragma unroll
					for(int i = 0; i < sizeof(output_valid_positions) / sizeof(unsigned int); ++i)
						output_valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					float input_local_buf[(DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1][WINDOW_WIDTH + BLOCK_WIDTH - 1];
					#pragma unroll
					for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? (WINDOW_HEIGHT + BLOCK_HEIGHT - 1) : 1); ++input_yy)
					{
						#pragma unroll
						for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
						{
							int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
							bool b_fit0 = ((input_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = input_elem_id + input_yy * input_sizes[0] + input_xx;
							input_local_buf[input_yy][input_xx] = tex1Dfetch<float>(input_tex, b_fit0 ? current_offset : -1);
						}
					}
					float output_local_buf[(DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1][WINDOW_WIDTH];
					#pragma unroll
					for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
					{
						#pragma unroll
						for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
						{
							int pos_total = output_yy * BLOCK_WIDTH + output_xx;
							bool b_fit0 = ((output_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = output_elem_id + output_yy * output_sizes[0] + output_xx;
							output_local_buf[output_yy][output_xx] = tex1Dfetch<float>(output_tex, b_fit0 ? current_offset : -1);
						}
					}

					#pragma unroll
					for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
					{
						#pragma unroll
						for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
						{
							#pragma unroll
							for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
							{
								#pragma unroll
								for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
								{
									local_gradients[weight_y][weight_x] += output_local_buf[output_yy][output_xx] * input_local_buf[output_yy + weight_y][output_xx + weight_x];
								}
							}
						}
					}

					input_elem_id += input_elem_count_per_entry;
					output_elem_id += output_elem_count_per_entry;
				} // for it

				int gradient_offset = feature_map_pair_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					gradient_offset = gradient_offset * window_sizes[i] + weight[i];
				float * base_gradients = gradients + gradient_offset;

				#pragma unroll
				for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
				{
					#pragma unroll
					for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
					{
						atomicAdd(base_gradients + (weight_y * WINDOW_WIDTH + weight_x), local_gradients[weight_y][weight_x]);
					}
				}
			} // if (in_bounds)
		}

		template<int DIMENSION_COUNT>
		__launch_bounds__(256, 3)
		__global__ void sparse_convolution_update_gradient_tex_generic_upd_kernel_kepler(
			float * __restrict gradients,
			cudaTextureObject_t output_tex,
			cudaTextureObject_t input_tex,
			const row_index_col_index_pair * __restrict row_index_col_index_pairs,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int entry_group_size,
			int input_elem_offset,
			int block_count_per_output_feature_map,
			int feature_map_pair_count,
			int input_elem_count_per_entry,
			int output_elem_count_per_entry,
			unsigned int dummy)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

			float weight[DIMENSION_COUNT];
			int weight_xyzw_feature_map_pair_id = blockIdx.y * blockDim.y + threadIdx.y;
			int feature_map_pair_id;
			bool weight_xyzw_feature_map_pair_fit;
			{
				int remainder = weight_xyzw_feature_map_pair_id / feature_map_pair_count;
				feature_map_pair_id = weight_xyzw_feature_map_pair_id - feature_map_pair_count * remainder;

				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / window_sizes[i];
					weight[i] = remainder - window_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				weight[DIMENSION_COUNT - 1] = remainder;
				weight_xyzw_feature_map_pair_fit = (weight[DIMENSION_COUNT - 1] < window_sizes[DIMENSION_COUNT - 1]);
			}

			int entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * entry_group_size;

			bool in_bounds = weight_xyzw_feature_map_pair_fit && (neuron_id < block_count_per_output_feature_map) && (entry_id < entry_count);
			if (in_bounds)
			{
				int xyzw_output[DIMENSION_COUNT];
				int remainder = neuron_id;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					int new_remainder = remainder / output_block_sizes[i];
					xyzw_output[i] = remainder - output_block_sizes[i] * new_remainder;
					remainder = new_remainder;
				}
				xyzw_output[DIMENSION_COUNT - 1] = remainder;
				xyzw_output[0] *= BLOCK_WIDTH;
				if (DIMENSION_COUNT > 1)
					xyzw_output[1] *= BLOCK_HEIGHT;

				int xyzw_input[DIMENSION_COUNT];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
				{
					xyzw_input[i] = xyzw_output[i] - left_zero_padding[i] + weight[i];
				}

				#pragma unroll
				for(int i = 2; i < DIMENSION_COUNT; ++i)
				{
					if ((unsigned int)xyzw_input[i] >= (unsigned int)input_sizes[i]);
						return;
				}

				row_index_col_index_pair rc = row_index_col_index_pairs[feature_map_pair_id];
				int output_feature_map_id = rc.row_index;
				int input_feature_map_id = rc.col_index;

				int input_elem_id = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw_input[i];
				input_elem_id += input_elem_offset;

				int output_elem_id = entry_id * output_feature_map_count + output_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					output_elem_id = output_elem_id * output_sizes[i] + xyzw_output[i];

				float local_gradient = 0.0F;

				unsigned int input_valid_positions[(((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1) * BLOCK_WIDTH + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(input_valid_positions) / sizeof(unsigned int); ++i)
					input_valid_positions[i] = 0;
				#pragma unroll
				for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++input_yy)
				{
					int input_y = input_yy + ((DIMENSION_COUNT > 1) ? xyzw_input[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)input_y < (unsigned int)input_sizes[1]) : true;
					#pragma unroll
					for(int input_xx = 0; input_xx < BLOCK_WIDTH; ++input_xx)
					{
						int input_x = input_xx + xyzw_input[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)input_x < (unsigned int)input_sizes[0]));
						if (b_fit0)
						{
							int pos_total = input_yy * BLOCK_WIDTH + input_xx;
							input_valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				unsigned int output_valid_positions[(((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1) * BLOCK_WIDTH + 31) / 32];
				#pragma unroll
				for(int i = 0; i < sizeof(output_valid_positions) / sizeof(unsigned int); ++i)
					output_valid_positions[i] = 0;
				#pragma unroll
				for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
				{
					int output_y = output_yy + ((DIMENSION_COUNT > 1) ? xyzw_output[1] : 0);
					bool b_fit1 = (DIMENSION_COUNT > 1) ? ((unsigned int)output_y < (unsigned int)output_sizes[1]) : true;
					#pragma unroll
					for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
					{
						int output_x = output_xx + xyzw_output[0];
						bool b_fit0 = (b_fit1 && ((unsigned int)output_x < (unsigned int)output_sizes[0]));
						if (b_fit0)
						{
							int pos_total = output_yy * BLOCK_WIDTH + output_xx;
							output_valid_positions[pos_total / 32] |= (1U << (pos_total & 31));
						}
					}
				}

				int it_count = min(entry_group_size, entry_count - entry_id);
				for(int it = 0; it < it_count; ++it)
				{
					#pragma unroll
					for(int i = 0; i < sizeof(input_valid_positions) / sizeof(unsigned int); ++i)
						input_valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					#pragma unroll
					for(int i = 0; i < sizeof(output_valid_positions) / sizeof(unsigned int); ++i)
						output_valid_positions[i] += dummy; // Hack to disable compiler putting each flag into its own 32bit register

					float input_local_buf[(DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1][BLOCK_WIDTH];
					#pragma unroll
					for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++input_yy)
					{
						#pragma unroll
						for(int input_xx = 0; input_xx < BLOCK_WIDTH; ++input_xx)
						{
							int pos_total = input_yy * BLOCK_WIDTH + input_xx;
							bool b_fit0 = ((input_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = input_elem_id + input_yy * input_sizes[0] + input_xx;
							input_local_buf[input_yy][input_xx] = tex1Dfetch<float>(input_tex, b_fit0 ? current_offset : -1);
						}
					}
					float output_local_buf[(DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1][BLOCK_WIDTH];
					#pragma unroll
					for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
					{
						#pragma unroll
						for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
						{
							int pos_total = output_yy * BLOCK_WIDTH + output_xx;
							bool b_fit0 = ((output_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = output_elem_id + output_yy * output_sizes[0] + output_xx;
							output_local_buf[output_yy][output_xx] = tex1Dfetch<float>(output_tex, b_fit0 ? current_offset : -1);
						}
					}

					#pragma unroll
					for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
					{
						#pragma unroll
						for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
						{
							local_gradient += output_local_buf[output_yy][output_xx] * input_local_buf[output_yy][output_xx];
						}
					}

					input_elem_id += input_elem_count_per_entry;
					output_elem_id += output_elem_count_per_entry;
				} // for it

				int gradient_offset = feature_map_pair_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					gradient_offset = gradient_offset * window_sizes[i] + weight[i];
				float * base_gradients = gradients + gradient_offset;

				atomicAdd(base_gradients, local_gradient);
			} // if (in_bounds)
		}

#define launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_tex_exact_blocked_upd_kernel_kepler<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, *data[0], *data_custom[0], *data_custom[1], *data[1], output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_elem_count_per_feature_map, entry_count, input_elem_offset, block_count_per_output_feature_map, weight_count_per_block, 0U);

#define launch_generic_kernel_const(dimension_count_const) \
	sparse_convolution_tex_generic_blocked_upd_kernel_kepler<dimension_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_neurons_buffer, input_tex, *data[0], *data_custom[0], *data_custom[1], *data[1], output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, input_elem_count_per_feature_map, entry_count, input_elem_offset, block_count_per_output_feature_map, weight_count_per_block);

#define launch_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const >= 1) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 2: \
			if (window_width_const >= 2) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 3: \
			if (window_width_const >= 3) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 4: \
			if (window_width_const >= 4) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 5: \
			if (window_width_const >= 5) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 6: \
			if (window_width_const >= 6) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 6); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		case 7: \
			if (window_width_const >= 7) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 7); } else { launch_generic_kernel_const(dimension_count_const); } \
			break; \
		default: \
			launch_generic_kernel_const(dimension_count_const); \
			break; \
		} \
	} \
	else \
	{ \
		launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); \
	}

#define launch_kernel(dimension_count_const, window_width, window_height) \
	switch (window_width) \
	{ \
	case 1: \
		launch_kernel_const_const(dimension_count_const, 1, window_height); \
		break; \
	case 2: \
		launch_kernel_const_const(dimension_count_const, 2, window_height); \
		break; \
	case 3: \
		launch_kernel_const_const(dimension_count_const, 3, window_height); \
		break; \
	case 4: \
		launch_kernel_const_const(dimension_count_const, 4, window_height); \
		break; \
	case 5: \
		launch_kernel_const_const(dimension_count_const, 5, window_height); \
		break; \
	case 6: \
		launch_kernel_const_const(dimension_count_const, 6, window_height); \
		break; \
	case 7: \
		launch_kernel_const_const(dimension_count_const, 7, window_height); \
		break; \
	default: \
		launch_generic_kernel_const(dimension_count_const); \
		break; \
	};

#define launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_backprop_tex_exact_blocked_upd_kernel_kepler<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, *data[0], row_index_weight_block_id_pairs, *data_custom[3], output_sizes, input_sizes, input_block_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, output_elem_count_per_feature_map, entry_count, block_count_per_input_feature_map, weight_count_per_block, 0U);

#define launch_backprop_generic_kernel_const(dimension_count_const) \
	sparse_convolution_backprop_tex_generic_blocked_upd_kernel_kepler<dimension_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, output_tex, *data[0], row_index_weight_block_id_pairs, *data_custom[3], output_sizes, input_sizes, input_block_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, output_elem_count_per_feature_map, entry_count, block_count_per_input_feature_map, weight_count_per_block);

#define launch_backprop_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const >= 1) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 2: \
			if (window_width_const >= 2) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 3: \
			if (window_width_const >= 3) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 4: \
			if (window_width_const >= 4) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 5: \
			if (window_width_const >= 5) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 6: \
			if (window_width_const >= 6) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 6); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		case 7: \
			if (window_width_const >= 7) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 7); } else { launch_backprop_generic_kernel_const(dimension_count_const); } \
			break; \
		default: \
			launch_backprop_generic_kernel_const(dimension_count_const); \
			break; \
		} \
	} \
	else \
	{ \
		launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); \
	}

#define launch_backprop_kernel(dimension_count_const, window_width, window_height) \
	switch (window_width) \
	{ \
	case 1: \
		launch_backprop_kernel_const_const(dimension_count_const, 1, window_height); \
		break; \
	case 2: \
		launch_backprop_kernel_const_const(dimension_count_const, 2, window_height); \
		break; \
	case 3: \
		launch_backprop_kernel_const_const(dimension_count_const, 3, window_height); \
		break; \
	case 4: \
		launch_backprop_kernel_const_const(dimension_count_const, 4, window_height); \
		break; \
	case 5: \
		launch_backprop_kernel_const_const(dimension_count_const, 5, window_height); \
		break; \
	case 6: \
		launch_backprop_kernel_const_const(dimension_count_const, 6, window_height); \
		break; \
	case 7: \
		launch_backprop_kernel_const_const(dimension_count_const, 7, window_height); \
		break; \
	default: \
		launch_backprop_generic_kernel_const(dimension_count_const); \
		break; \
	};


#define launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_update_gradient_tex_exact_blocked_upd_kernel_kepler<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*gradient[0], output_tex, input_tex, row_index_col_index_pairs, output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count, entry_group_size_and_count.first, input_elem_offset, block_count_per_output_feature_map, feature_map_connection_count, input_elem_count_per_entry, output_elem_count_per_entry, 0U);

#define launch_update_gradient_generic_kernel_const(dimension_count_const) \
	sparse_convolution_update_gradient_tex_generic_upd_kernel_kepler<dimension_count_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*gradient[0], output_tex, input_tex, row_index_col_index_pairs, output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count, entry_count, entry_group_size_and_count.first, input_elem_offset, block_count_per_output_feature_map, feature_map_connection_count, input_elem_count_per_entry, output_elem_count_per_entry, 0U);

#define launch_update_gradient_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const >= 1) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 2: \
			if (window_width_const >= 2) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 3: \
			if (window_width_const >= 3) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 4: \
			if (window_width_const >= 4) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 5: \
			if (window_width_const >= 5) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 6: \
			if (window_width_const >= 6) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 6); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		case 7: \
			if (window_width_const >= 7) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 7); } else { launch_update_gradient_generic_kernel_const(dimension_count_const); } \
			break; \
		default: \
			launch_update_gradient_generic_kernel_const(dimension_count_const); \
			break; \
		} \
	} \
	else \
	{ \
		launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); \
	}

#define launch_update_gradient_kernel(dimension_count_const, window_width, window_height) \
	switch (window_width) \
	{ \
	case 1: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 1, window_height); \
		break; \
	case 2: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 2, window_height); \
		break; \
	case 3: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 3, window_height); \
		break; \
	case 4: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 4, window_height); \
		break; \
	case 5: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 5, window_height); \
		break; \
	case 6: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 6, window_height); \
		break; \
	case 7: \
		launch_update_gradient_kernel_const_const(dimension_count_const, 7, window_height); \
		break; \
	default: \
		launch_update_gradient_generic_kernel_const(dimension_count_const); \
		break; \
	};



		template<int dimension_count>
		class sparse_convolution_layer_updater_cuda_kepler : public layer_updater_cuda
		{
		public:
			sparse_convolution_layer_updater_cuda_kepler()
			{
			}

			virtual ~sparse_convolution_layer_updater_cuda_kepler()
			{
			}

			virtual void enqueue_test(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count)
			{
				cuda_texture input_tex(input_neurons_buffer);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count_per_output_feature_map * output_configuration_specific.feature_map_count,
					entry_count,
					1);

				int input_elem_offset = offset_input_entry_id * input_elem_count_per_entry;

				launch_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
			}

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count)
			{
				if (!backprop_required)
					throw neural_network_exception("sparse_convolution_layer_updater_cuda_kepler is not configured to do backprop but requested to");

				const row_index_weight_block_id_pair * row_index_weight_block_id_pairs = (row_index_weight_block_id_pair *)((void *)(*data_custom[2]));

				cuda_texture output_tex(output_errors_buffer);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count_per_input_feature_map * input_configuration_specific.feature_map_count,
					entry_count,
					1);

				launch_backprop_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
			}

			virtual void enqueue_update_weights(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count)
			{
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
					sparse_convolution_update_biases_upd_kernel_kepler<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
						*gradient[1],
						*output_errors_buffer,
						block_size,
						output_elem_count_per_feature_map,
						output_configuration_specific.feature_map_count,
						entry_count);
				}

				// Update weights
				{
					const row_index_col_index_pair * row_index_col_index_pairs = (row_index_col_index_pair *)((void *)(*data_custom[4]));

					cuda_texture input_tex(input_neurons_buffer);
					cuda_texture output_tex(output_errors_buffer);

					bool exact_kernel;
					if (dimension_count > 1)
					{
						exact_kernel = ((window_sizes[1] <= MAX_EXACT_GRADIENT_UPDATE_WINDOW_WIDTH_HEIGHT) && (window_sizes[0] <= window_sizes[1]));
					}
					else
					{
						exact_kernel = (window_sizes[0] <= MAX_EXACT_GRADIENT_UPDATE_WINDOW_WIDTH_HEIGHT);
					}

					int update_weight_count = 1;
					for(int i = (exact_kernel ? 2 : 0); i < dimension_count; ++i)
						update_weight_count *= window_sizes[i];

					std::pair<int, int> entry_group_size_and_count = get_entry_group_size_and_count(entry_count);

					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						block_count_per_output_feature_map,
						feature_map_connection_count * update_weight_count,
						entry_group_size_and_count.second);

					int input_elem_offset = offset_input_entry_id * input_elem_count_per_entry;

					launch_update_gradient_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
				}
			}

		protected:
			virtual bool is_in_place_backprop() const
			{
				return false;
			}

			virtual void updater_configured()
			{
				nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

				feature_map_connection_count = layer_derived->feature_map_connection_count;

				block_count_per_output_feature_map = 1;
				block_count_per_input_feature_map = 1;
				weight_count_per_block = 1;
				for(int i = 0; i < dimension_count; ++i)
				{
					window_sizes[i] = layer_derived->window_sizes[i];
					input_sizes[i] = input_configuration_specific.dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];
					left_zero_padding[i] = layer_derived->left_zero_padding[i];

					switch (i)
					{
					case 0:
						output_block_sizes[i] = (output_sizes[i] + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
						input_block_sizes[i] = (input_sizes[i] + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
						break;
					case 1:
						output_block_sizes[i] = (output_sizes[i] + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
						input_block_sizes[i] = (input_sizes[i] + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
						break;
					default:
						output_block_sizes[i] = output_sizes[i];
						input_block_sizes[i] = input_sizes[i];
						break;
					}

					block_count_per_output_feature_map *= output_block_sizes[i];
					block_count_per_input_feature_map *= input_block_sizes[i];
					weight_count_per_block *= window_sizes[i];
				}
			}

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const
			{
				std::vector<unsigned int> res;
				res.push_back(input_configuration_specific.get_neuron_count());
				return res;
			}

			virtual std::vector<cuda_linear_buffer_device_smart_ptr> get_data_custom(const_layer_data_custom_smart_ptr host_data_custom) const
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> res;

				const std::vector<int>& column_indices = host_data_custom->at(0);
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*column_indices.begin()),
					column_indices.size() * sizeof(int))));
				const std::vector<int>& row_ptrs = host_data_custom->at(1);
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*row_ptrs.begin()),
					row_ptrs.size() * sizeof(int))));

				std::vector<std::vector<row_index_weight_block_id_pair> > column_row_index_weight_block_id_pair_list(input_configuration_specific.feature_map_count);

				for(int output_feature_map_id = 0; output_feature_map_id < static_cast<int>(output_configuration_specific.feature_map_count); ++output_feature_map_id)
				{
					row_index_weight_block_id_pair new_elem;
					new_elem.row_index = output_feature_map_id;

					int start_column_index = row_ptrs[output_feature_map_id];
					int end_column_index = row_ptrs[output_feature_map_id + 1];
					for(int nnz_index = start_column_index; nnz_index < end_column_index; ++nnz_index)
					{
						int input_feature_map_id = column_indices[nnz_index];
						new_elem.weight_block_id = nnz_index;
						column_row_index_weight_block_id_pair_list[input_feature_map_id].push_back(new_elem);
					}
				}

				std::vector<row_index_weight_block_id_pair> row_index_weight_block_id_pairs(column_indices.size());
				std::vector<int> col_ptrs(input_configuration_specific.feature_map_count + 1);

				int current_row_offset = 0;
				for(int input_feature_map_id = 0; input_feature_map_id < static_cast<int>(input_configuration_specific.feature_map_count); ++input_feature_map_id)
				{
					col_ptrs[input_feature_map_id] = current_row_offset;
					std::copy(
						column_row_index_weight_block_id_pair_list[input_feature_map_id].begin(),
						column_row_index_weight_block_id_pair_list[input_feature_map_id].end(),
						row_index_weight_block_id_pairs.begin() + current_row_offset);

					current_row_offset += static_cast<int>(column_row_index_weight_block_id_pair_list[input_feature_map_id].size());
				}
				col_ptrs[input_configuration_specific.feature_map_count] = current_row_offset;

				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*row_index_weight_block_id_pairs.begin()),
					row_index_weight_block_id_pairs.size() * sizeof(row_index_weight_block_id_pair))));
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*col_ptrs.begin()),
					col_ptrs.size() * sizeof(int))));

				int current_elem_id = 0;
				std::vector<row_index_col_index_pair> row_index_col_index_pairs(feature_map_connection_count);
				for(int output_feature_map_id = 0; output_feature_map_id < static_cast<int>(output_configuration_specific.feature_map_count); ++output_feature_map_id)
				{
					int start_column_index = row_ptrs[output_feature_map_id];
					int end_column_index = row_ptrs[output_feature_map_id + 1];
					for(int nnz_index = start_column_index; nnz_index < end_column_index; ++nnz_index)
					{
						int input_feature_map_id = column_indices[nnz_index];
						row_index_col_index_pairs[current_elem_id].row_index = output_feature_map_id;
						row_index_col_index_pairs[current_elem_id].col_index = input_feature_map_id;
						++current_elem_id;
					}
				}

				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*row_index_col_index_pairs.begin()),
					row_index_col_index_pairs.size() * sizeof(row_index_col_index_pair))));

				return res;
			}

		private:
			static int get_bias_update_block_size(int entry_count)
			{
				int block_size = std::min(std::max(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
				return block_size;
			}

			std::pair<int, int> get_entry_group_size_and_count(int entry_count) const
			{
				int group_count = (entry_count + preferred_entry_group_size - 1) / preferred_entry_group_size;
				int group_size = (entry_count + group_count - 1) / group_count;

				return std::make_pair(group_size, group_count);
			}

		private:
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> output_block_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> input_block_sizes;
			array_by_val<int, dimension_count> window_sizes;
			array_by_val<int, dimension_count> left_zero_padding;
			int block_count_per_output_feature_map;
			int block_count_per_input_feature_map;
			int weight_count_per_block;
			int feature_map_connection_count;

			static const int preferred_entry_group_size = 8;
		};
	}
}
