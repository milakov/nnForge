/*
 *  Copyright 2011-2016 Maxim Milakov
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
#include "neural_network_cudnn_exception.h"
#include "int_fastdiv.h"

#include "../sparse_convolution_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include <cudnn.h>

#define MAX_DIMENSION_COUNT 3

#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4

#define MAX_EXACT_GRADIENT_UPDATE_WINDOW_WIDTH_HEIGHT 5

#define OUTPUT_FEATURE_MAP_BLOCK 2

namespace nnforge
{
	namespace cuda
	{
		struct __align__(8) row_index_weight_block_id_pair
		{
			int row_index;
			int weight_block_id;
		};

		struct col_index_row_indices_weight_indices
		{
			int col_index;
			int row_indices[OUTPUT_FEATURE_MAP_BLOCK];
			int weight_indices[OUTPUT_FEATURE_MAP_BLOCK];
		};

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int WINDOW_HEIGHT>
		__global__ void sparse_convolution_exact_blocked_upd_kernel(
			float * __restrict output,
			const float * __restrict input,
			const float * __restrict weights,
			const int * __restrict column_indices,
			const int * __restrict row_ptrs,
			const float * __restrict biases,
			bool bias_exists,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int input_elem_count_per_feature_map,
			int entry_count,
			int block_count_per_feature_map,
			int weight_count_per_block,
			unsigned int dummy)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (output_feature_map_id < output_feature_map_count) && (neuron_id < block_count_per_feature_map);
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

				float sums[(DIMENSION_COUNT > 1 ? BLOCK_HEIGHT : 1)][BLOCK_WIDTH];

				float bias = bias_exists ? biases[output_feature_map_id] : 0.0F;
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

					for(int input_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); input_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++input_z)
					{
						bool b_fit2 = (DIMENSION_COUNT > 2) ? ((unsigned int)input_z < (unsigned int)input_sizes[2]) : true;

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
								input_local_buf[input_yy][input_xx] = b_fit0 ? __load_nc(input + current_offset) : 0.0F;
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
		__launch_bounds__(256, 3)
		__global__ void sparse_convolution_backprop_exact_blocked_upd_kernel(
			float * __restrict input,
			const float * __restrict output_errors,
			const float * __restrict weights,
			const row_index_weight_block_id_pair * __restrict row_index_weight_block_id_pairs,
			const int * __restrict col_ptrs,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> input_block_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int output_elem_count_per_feature_map,
			int entry_count,
			int block_count_per_feature_map,
			int weight_count_per_block,
			unsigned int dummy,
			bool add_update_to_destination)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int input_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (input_feature_map_id < input_feature_map_count) && (neuron_id < block_count_per_feature_map);
			if (in_bounds)
			{
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
				#pragma unroll
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

					for(int output_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); output_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++output_z)
					{
						bool b_fit2 = (DIMENSION_COUNT > 2) ? ((unsigned int)output_z < (unsigned int)output_sizes[2]) : true;

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
								output_local_buf[output_yy][output_xx] = b_fit0 ? __load_nc(output_errors + current_offset) : 0.0F;
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
				} // for nnz_index

				int input_offset = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_offset = input_offset * input_sizes[i] + xyzw_input[i];

				if (add_update_to_destination)
				{
					#pragma unroll
					for(int pos_y = 0; pos_y < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++pos_y)
					{
						if ((DIMENSION_COUNT > 1) ? (pos_y < input_sizes[1] - xyzw_input[1]) : true)
						{
							float old_values[BLOCK_WIDTH];
							for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
							{
								if (pos_x < input_sizes[0] - xyzw_input[0])
								{
									old_values[pos_x] = input[input_offset + pos_x];
								}
							}
							#pragma unroll
							for(int pos_x = 0; pos_x < BLOCK_WIDTH; ++pos_x)
							{
								if (pos_x < input_sizes[0] - xyzw_input[0])
								{
									input[input_offset + pos_x] = sums[pos_y][pos_x] + old_values[pos_x];
								}
							}
						}
						input_offset += input_sizes[0];
					}
				}
				else
				{
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
				}
			} // if (in_bounds)
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int WINDOW_HEIGHT>
		__launch_bounds__(256, 4)
		__global__ void sparse_convolution_update_gradient_exact_blocked_upd_kernel(
			float * __restrict gradients,
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			const col_index_row_indices_weight_indices * __restrict col_index_row_indices_weight_indices_buffer,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> output_block_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int_fastdiv, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count,
			int output_feature_map_count,
			int entry_count,
			int entry_group_size,
			int block_count_per_output_feature_map,
			int_fastdiv feature_map_pair_count,
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

				int input_feature_map_id = col_index_row_indices_weight_indices_buffer[feature_map_pair_id].col_index;
				int output_feature_map_ids[OUTPUT_FEATURE_MAP_BLOCK];
				#pragma unroll
				for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
					output_feature_map_ids[block_id] = col_index_row_indices_weight_indices_buffer[feature_map_pair_id].row_indices[block_id];

				int input_elem_id = entry_id * input_feature_map_count + input_feature_map_id;
				#pragma unroll
				for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
					input_elem_id = input_elem_id * input_sizes[i] + xyzw_input[i];

				int output_elem_ids[OUTPUT_FEATURE_MAP_BLOCK];
				#pragma unroll
				for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
				{
					output_elem_ids[block_id] = entry_id * output_feature_map_count + output_feature_map_ids[block_id];
					#pragma unroll
					for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
						output_elem_ids[block_id] = output_elem_ids[block_id] * output_sizes[i] + xyzw_output[i];
				}

				float local_gradients[OUTPUT_FEATURE_MAP_BLOCK][WINDOW_HEIGHT][WINDOW_WIDTH];
				#pragma unroll
				for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
					#pragma unroll
					for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
						#pragma unroll
						for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
							local_gradients[block_id][weight_y][weight_x] = 0.0F;

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

					float input_local_buf[(DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1][WINDOW_WIDTH + BLOCK_WIDTH - 1];
					#pragma unroll
					for(int input_yy = 0; input_yy < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT - 1 : 0); ++input_yy)
					{
						#pragma unroll
						for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
						{
							int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
							bool b_fit0 = ((input_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = input_elem_id + input_yy * input_sizes[0] + input_xx;
							input_local_buf[input_yy][input_xx] = b_fit0 ? __load_nc(input_neurons + current_offset) : 0.0F;
						}
					}

					#pragma unroll
					for(int output_yy = 0; output_yy < ((DIMENSION_COUNT > 1) ? BLOCK_HEIGHT : 1); ++output_yy)
					{
						int base_input_y = output_yy % ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1);
						int dest_local_input_y = (base_input_y + ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1) - 1) % ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1);
						int input_yy = output_yy + (((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1) - 1);
						#pragma unroll
						for(int input_xx = 0; input_xx < (WINDOW_WIDTH + BLOCK_WIDTH - 1); ++input_xx)
						{
							int pos_total = input_yy * (WINDOW_WIDTH + BLOCK_WIDTH - 1) + input_xx;
							bool b_fit0 = ((input_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							int current_offset = input_elem_id + input_yy * input_sizes[0] + input_xx;
							input_local_buf[dest_local_input_y][input_xx] = b_fit0 ? __load_nc(input_neurons + current_offset) : 0.0F;
						}

						float output_local_buf[OUTPUT_FEATURE_MAP_BLOCK][BLOCK_WIDTH];
						#pragma unroll
						for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
						{
							int pos_total = output_yy * BLOCK_WIDTH + output_xx;
							bool b_fit0 = ((output_valid_positions[pos_total / 32] & (1U << (pos_total & 31))) != 0);
							#pragma unroll
							for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
								output_local_buf[block_id][output_xx] = 0.0F;
							if (b_fit0)
							{
								#pragma unroll
								for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
								{
									int current_offset = output_elem_ids[block_id] + output_yy * output_sizes[0] + output_xx;
									output_local_buf[block_id][output_xx] = __load_nc(output_errors + current_offset);
								}
							}
						}

						#pragma unroll
						for(int output_xx = 0; output_xx < BLOCK_WIDTH; ++output_xx)
						{
							#pragma unroll
							for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
							{
								#pragma unroll
								for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
								{
									#pragma unroll
									for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
									{
										local_gradients[block_id][weight_y][weight_x] += output_local_buf[block_id][output_xx] * input_local_buf[(base_input_y + weight_y) % ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1)][output_xx + weight_x];
									}
								}
							}
						}
					}

					input_elem_id += input_elem_count_per_entry;
					#pragma unroll
					for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
						output_elem_ids[block_id] += output_elem_count_per_entry;
				} // for it


				int gradient_offsets[OUTPUT_FEATURE_MAP_BLOCK];
				#pragma unroll
				for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
					gradient_offsets[block_id] = col_index_row_indices_weight_indices_buffer[feature_map_pair_id].weight_indices[block_id];

				#pragma unroll
				for(int block_id = 0; block_id < OUTPUT_FEATURE_MAP_BLOCK; ++block_id)
				{
					if (gradient_offsets[block_id] >= 0)
					{
						#pragma unroll
						for(int i = DIMENSION_COUNT - 1; i >= 0; --i)
							gradient_offsets[block_id] = gradient_offsets[block_id] * window_sizes[i] + weight[i];
						float * base_gradients = gradients + gradient_offsets[block_id];

						#pragma unroll
						for(int weight_y = 0; weight_y < ((DIMENSION_COUNT > 1) ? WINDOW_HEIGHT : 1); ++weight_y)
						{
							#pragma unroll
							for(int weight_x = 0; weight_x < WINDOW_WIDTH; ++weight_x)
							{
								atomicAdd(base_gradients + (weight_y * WINDOW_WIDTH + weight_x), local_gradients[block_id][weight_y][weight_x]);
							}
						}
					}
				}
			} // if (in_bounds)
		}

#define launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_exact_blocked_upd_kernel<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_buffer, *input_buffers[0], *data[0], *data_custom[0], *data_custom[1], bias ? *data[1] : (const float *)0, bias, output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count, input_elem_count_per_feature_map_list[0], entry_count, block_count_per_output_feature_map, weight_count_per_block, 0U);

#define launch_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const <= 1) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 2: \
			if (window_width_const <= 2) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 3: \
			if (window_width_const <= 3) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 4: \
			if (window_width_const <= 4) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 5: \
			if (window_width_const <= 5) { launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		default: \
			throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
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
	default: \
		throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
	};

#define launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_backprop_exact_blocked_upd_kernel<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*input_errors_buffer, *output_errors_buffer, *data[0], row_index_weight_block_id_pairs, *data_custom[3], output_sizes, input_sizes, input_block_sizes, window_sizes, left_zero_padding, input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count, output_elem_count_per_feature_map, entry_count, block_count_per_input_feature_map, weight_count_per_block, 0U, add_update_to_destination);

#define launch_backprop_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const <= 1) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 2: \
			if (window_width_const <= 2) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 3: \
			if (window_width_const <= 3) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 4: \
			if (window_width_const <= 4) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 5: \
			if (window_width_const <= 5) { launch_backprop_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		default: \
			throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
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
	default: \
		throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
	};


#define launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, window_height_const) \
	sparse_convolution_update_gradient_exact_blocked_upd_kernel<dimension_count_const,window_width_const,window_height_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*gradient[0], *output_errors_buffer, *input_neurons_buffers[0], crw, output_sizes, output_block_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count, entry_count, entry_group_size_and_count.first, block_count_per_output_feature_map, crw_count, input_elem_count_per_entry_list[0], output_elem_count_per_entry, 0U);

#define launch_update_gradient_kernel_const_const(dimension_count_const, window_width_const, window_height) \
	if (dimension_count_const > 1) \
	{ \
		switch (window_height) \
		{ \
		case 1: \
			if (window_width_const <= 1) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 1); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 2: \
			if (window_width_const <= 2) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 2); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 3: \
			if (window_width_const <= 3) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 3); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 4: \
			if (window_width_const <= 4) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 4); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		case 5: \
			if (window_width_const <= 5) { launch_update_gradient_exact_kernel_const_const_const(dimension_count_const, window_width_const, 5); } else { throw neural_network_exception("Unsupported config for sparse convolutional layer"); } \
			break; \
		default: \
			throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
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
	default: \
		throw neural_network_exception("Unsupported config for sparse convolutional layer"); \
	};

		template<int dimension_count>
		class sparse_convolution_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			sparse_convolution_layer_updater_cuda()
				: output_data_desc(0)
				, bias_desc(0)
			{
				cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
				cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
			}

			virtual ~sparse_convolution_layer_updater_cuda()
			{
				cudnnDestroyTensorDescriptor(output_data_desc);
				cudnnDestroyTensorDescriptor(bias_desc);
			}

			virtual void enqueue_forward_propagation(
				cudaStream_t stream_id,
				cuda_linear_buffer_device::ptr output_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::ptr temporary_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
				unsigned int entry_count)
			{
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count_per_output_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count);

				launch_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
			}

			virtual void enqueue_backward_data_propagation(
				cudaStream_t stream_id,
				unsigned int input_index,
				cuda_linear_buffer_device::ptr input_errors_buffer,
				cuda_linear_buffer_device::const_ptr output_errors_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
				cuda_linear_buffer_device::const_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
				cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
				bool add_update_to_destination,
				unsigned int entry_count)
			{
				const row_index_weight_block_id_pair * row_index_weight_block_id_pairs = (row_index_weight_block_id_pair *)((const void *)(*data_custom[2]));

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					block_count_per_input_feature_map,
					input_configuration_specific_list[0].feature_map_count,
					entry_count);

				launch_backprop_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
			}

			virtual void enqueue_backward_weights_propagation(
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::ptr>& gradient,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
				cuda_linear_buffer_device::const_ptr output_errors_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
				cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
				unsigned int entry_count)
			{
				// Update weights
				{
					const col_index_row_indices_weight_indices * crw = (col_index_row_indices_weight_indices *)((const void *)(*data_custom[4]));

					int update_weight_count = 1;
					for(int i = 2; i < dimension_count; ++i)
						update_weight_count *= window_sizes[i];

					std::pair<int, int> entry_group_size_and_count = get_entry_group_size_and_count(entry_count);

					std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
						*cuda_config,
						block_count_per_output_feature_map,
						crw_count * update_weight_count,
						entry_group_size_and_count.second);

					launch_update_gradient_kernel(dimension_count, window_sizes[0], ((dimension_count > 1) ? window_sizes[1] : 1));
				}

				// Update bias
				if (bias)
				{
					cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
					cudnn_safe_call(cudnnSetTensor4dDescriptor(
						output_data_desc,
						CUDNN_TENSOR_NCHW,
						CUDNN_DATA_FLOAT,
						entry_count,
						output_configuration_specific.feature_map_count,
						1,
						output_elem_count_per_feature_map));

					float alpha = 1.0F;
					float beta = 1.0F;
					cudnn_safe_call(cudnnConvolutionBackwardBias(
						cuda_config->get_cudnn_handle(),
						&alpha,
						output_data_desc,
						*output_errors_buffer,
						&beta,
						bias_desc,
						*gradient[1]));
				}
			}

			virtual std::vector<cuda_linear_buffer_device::const_ptr> set_get_data_custom(layer_data_custom::const_ptr host_data)
			{
				std::vector<cuda_linear_buffer_device::const_ptr> res;

				const std::vector<int>& column_indices = host_data->at(0);
				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(*column_indices.begin()),
					column_indices.size() * sizeof(int))));
				const std::vector<int>& row_ptrs = host_data->at(1);
				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(*row_ptrs.begin()),
					row_ptrs.size() * sizeof(int))));

				std::vector<std::vector<row_index_weight_block_id_pair> > column_row_index_weight_block_id_pair_list(input_configuration_specific_list[0].feature_map_count);

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
				std::vector<int> col_ptrs(input_configuration_specific_list[0].feature_map_count + 1);

				int current_row_offset = 0;
				for(int input_feature_map_id = 0; input_feature_map_id < static_cast<int>(input_configuration_specific_list[0].feature_map_count); ++input_feature_map_id)
				{
					col_ptrs[input_feature_map_id] = current_row_offset;
					std::copy(
						column_row_index_weight_block_id_pair_list[input_feature_map_id].begin(),
						column_row_index_weight_block_id_pair_list[input_feature_map_id].end(),
						row_index_weight_block_id_pairs.begin() + current_row_offset);

					current_row_offset += static_cast<int>(column_row_index_weight_block_id_pair_list[input_feature_map_id].size());
				}
				col_ptrs[input_configuration_specific_list[0].feature_map_count] = current_row_offset;

				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(*row_index_weight_block_id_pairs.begin()),
					row_index_weight_block_id_pairs.size() * sizeof(row_index_weight_block_id_pair))));
				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(*col_ptrs.begin()),
					col_ptrs.size() * sizeof(int))));

				std::vector<col_index_row_indices_weight_indices> crw;
				for(int input_feature_map_id = 0; input_feature_map_id < static_cast<int>(input_configuration_specific_list[0].feature_map_count); ++input_feature_map_id)
				{
					const std::vector<row_index_weight_block_id_pair>& rw_list = column_row_index_weight_block_id_pair_list[input_feature_map_id];
					int src_index = 0;
					for(int index = 0; index < (rw_list.size() + (OUTPUT_FEATURE_MAP_BLOCK - 1)) / OUTPUT_FEATURE_MAP_BLOCK; ++index)
					{
						col_index_row_indices_weight_indices new_item;
						new_item.col_index = input_feature_map_id;
						new_item.row_indices[0] = rw_list[src_index].row_index;
						new_item.weight_indices[0] = rw_list[src_index].weight_block_id;
						++src_index;
						for(int i = 1; i < OUTPUT_FEATURE_MAP_BLOCK; ++i, ++src_index)
						{
							if (src_index < rw_list.size())
							{
								new_item.row_indices[i] = rw_list[src_index].row_index;
								new_item.weight_indices[i] = rw_list[src_index].weight_block_id;
							}
							else
							{
								new_item.row_indices[i] = new_item.row_indices[i - 1];
								new_item.weight_indices[i] = -1;
							}
						}
						crw.push_back(new_item);
					}
				}
				crw_count = static_cast<int>(crw.size());

				res.push_back(cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(crw[0]),
					crw.size() * sizeof(col_index_row_indices_weight_indices))));

				return res;
			}

			virtual bool is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
			{
				return false;
			}

			virtual bool is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
			{
				return false;
			}

			virtual bool is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
			{
				return true;
			}

		protected:
			virtual void updater_configured()
			{
				nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

				bias = layer_derived->bias;

				block_count_per_output_feature_map = 1;
				block_count_per_input_feature_map = 1;
				weight_count_per_block = 1;
				for(int i = 0; i < dimension_count; ++i)
				{
					window_sizes[i] = layer_derived->window_sizes[i];
					input_sizes[i] = input_configuration_specific_list[0].dimension_sizes[i];
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

				cudnn_safe_call(cudnnSetTensor4dDescriptor(
					bias_desc,
					CUDNN_TENSOR_NCHW,
					CUDNN_DATA_FLOAT,
					1,
					output_configuration_specific.feature_map_count,
					1,
					1));
			}

		private:
			std::pair<int, int> get_entry_group_size_and_count(int entry_count) const
			{
				int group_count = (entry_count + preferred_entry_group_size - 1) / preferred_entry_group_size;
				int group_size = (entry_count + group_count - 1) / group_count;

				return std::make_pair(group_size, group_count);
			}

		private:
			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int_fastdiv, dimension_count> output_block_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int_fastdiv, dimension_count> input_block_sizes;
			array_by_val<int_fastdiv, dimension_count> window_sizes;
			array_by_val<int, dimension_count> left_zero_padding;
			int block_count_per_output_feature_map;
			int block_count_per_input_feature_map;
			int weight_count_per_block;
			int_fastdiv crw_count;
			bool bias;

			static const int preferred_entry_group_size = 8;

			cudnnTensorDescriptor_t output_data_desc;
			cudnnTensorDescriptor_t bias_desc;
		};
	}
}
