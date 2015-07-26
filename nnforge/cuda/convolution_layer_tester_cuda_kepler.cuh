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

#include "layer_tester_cuda.h"

#include <cuda_runtime.h>

#include <boost/format.hpp>

#include "util_cuda.h"
#include "cuda_texture.h"
#include "neural_network_cuda_exception.h"
#include "packed_config.h"
#include "sequential_curve.h"

#include "../convolution_layer.h"
#include "../nn_types.h"

#define FEATURE_MAP_BLOCK_SIZE 4
#define MAX_BLOCK_SIZE 5
#define MAX_DIMENSION_COUNT 4

namespace nnforge
{
	namespace cuda
	{
		template<int DIMENSION_COUNT, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_tex_generic_blocked_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int xyzw_output[DIMENSION_COUNT];
			int xyzw[DIMENSION_COUNT];
			xyzw_output[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			xyzw[0] = xyzw_output[0] - left_zero_padding[0];
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw_output[0] < output_sizes[0]) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = input_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_output_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					xyzw_output[i + 1] = conf.get_val(i);
					xyzw[i + 1] = xyzw_output[i + 1] - left_zero_padding[i + 1];
				}
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
					for(int input_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); input_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)input_w < (unsigned int)input_sizes[3]) : true;
						for(int input_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); input_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)input_z < (unsigned int)input_sizes[2])) : true;
							for(int input_y = (DIMENSION_COUNT > 1 ? xyzw[1] : 0); input_y < (DIMENSION_COUNT > 1 ? xyzw[1] + window_sizes[1] : 1); ++input_y)
							{
								bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && ((unsigned int)input_y < (unsigned int)input_sizes[1])) : true;

								#pragma unroll 4
								for(int input_x = xyzw[0]; input_x < xyzw[0] + window_sizes[0]; ++input_x)
								{
									float2 weight_list[FEATURE_MAP_BLOCK_SIZE];
									#pragma unroll
									for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										weight_list[i] = tex1Dfetch<float2>(weights_tex, weights_offset + weight_count_per_output_feature_map * i);
									#pragma unroll
									for(int j = 0; j < BLOCK_SIZE; ++j)
									{
										int input_x_total = input_x + j;
										bool b_fit0 = b_fit1 && ((unsigned int)input_x_total < (unsigned int)input_sizes[0]);
										float2 inp = tex1Dfetch<float2>(input_tex, b_fit0 ? (input_elem_id - xyzw[0] + input_x_total) : -1); 

										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									weights_offset++;
								} // for input_x
								input_elem_id += input_sizes[0];
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
					output_offset = output_offset * output_sizes[i] + xyzw_output[i];
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
							if (j < output_sizes[0] - xyzw_output[0])
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

		template<int DIMENSION_COUNT, int WINDOW_WIDTH, int BLOCK_SIZE>
		__launch_bounds__(256, 4)
		__global__ void convolution_tex_exact_blocked_kernel_kepler(
			float * __restrict output,
			cudaTextureObject_t input_tex,
			cudaTextureObject_t weights_tex,
			const float * __restrict biases,
			const packed_config<DIMENSION_COUNT> * __restrict packed_config_list,
			array_by_val<int, DIMENSION_COUNT> output_sizes,
			array_by_val<int, DIMENSION_COUNT> input_sizes,
			array_by_val<int, DIMENSION_COUNT> window_sizes,
			array_by_val<int, DIMENSION_COUNT> left_zero_padding,
			int input_feature_map_count_striped,
			int output_feature_map_count,
			int entry_count,
			int packed_config_count)
		{
			int xyzw_output[DIMENSION_COUNT];
			int xyzw[DIMENSION_COUNT];
			xyzw_output[0] = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
			xyzw[0] = xyzw_output[0] - left_zero_padding[0];
			int packed_config_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			bool in_bounds = (entry_id < entry_count) && (xyzw_output[0] < output_sizes[0]) && (packed_config_id < packed_config_count);
			if (in_bounds)
			{
				int weight_count_per_output_feature_map = input_feature_map_count_striped;
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT; ++i)
					weight_count_per_output_feature_map *= window_sizes[i];
				packed_config<DIMENSION_COUNT> conf = packed_config_list[packed_config_id];
				#pragma unroll
				for(int i = 0; i < DIMENSION_COUNT - 1; ++i)
				{
					xyzw_output[i + 1] = conf.get_val(i);
					xyzw[i + 1] = xyzw_output[i + 1] - left_zero_padding[i + 1];
				}
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
					for(int input_w = (DIMENSION_COUNT > 3 ? xyzw[3] : 0); input_w < (DIMENSION_COUNT > 3 ? xyzw[3] + window_sizes[3] : 1); ++input_w)
					{
						bool b_fit3 = (DIMENSION_COUNT > 3) ? ((unsigned int)input_w < (unsigned int)input_sizes[3]) : true;
						for(int input_z = (DIMENSION_COUNT > 2 ? xyzw[2] : 0); input_z < (DIMENSION_COUNT > 2 ? xyzw[2] + window_sizes[2] : 1); ++input_z)
						{
							bool b_fit2 = (DIMENSION_COUNT > 2) ? (b_fit3 && ((unsigned int)input_z < (unsigned int)input_sizes[2])) : true;
							for(int input_y = (DIMENSION_COUNT > 1 ? xyzw[1] : 0); input_y < (DIMENSION_COUNT > 1 ? xyzw[1] + window_sizes[1] : 1); ++input_y)
							{
								bool b_fit1 = (DIMENSION_COUNT > 1) ? (b_fit2 && ((unsigned int)input_y < (unsigned int)input_sizes[1])) : true;

								float2 input_vals[BLOCK_SIZE + WINDOW_WIDTH - 1];
								int input_x = xyzw[0];
								#pragma unroll
								for(int i = 0; i < BLOCK_SIZE + WINDOW_WIDTH - 1; ++i, ++input_x)
								{
									bool b_fit0 = b_fit1 && ((unsigned int)input_x < (unsigned int)input_sizes[0]);
									input_vals[i] = tex1Dfetch<float2>(input_tex, b_fit0 ? (input_elem_id + i) : -1);
								}

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
										float2 inp = input_vals[input_x + j]; 
										#pragma unroll
										for(int i = 0; i < FEATURE_MAP_BLOCK_SIZE; ++i)
										{
											sums[i * BLOCK_SIZE + j] += inp.x * weight_list[i].x;
											sums[i * BLOCK_SIZE + j] += inp.y * weight_list[i].y;
										}
									}
									weights_offset++;
								} // input_x
								input_elem_id += input_sizes[0];
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
					output_offset = output_offset * output_sizes[i] + xyzw_output[i];
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
							if (j < output_sizes[0] - xyzw_output[0])
								base_output[j + output_neuron_count_per_feature_map * i] = sums[i * BLOCK_SIZE + j];
						}
					}
				}
			}
		}

#define launch_exact_kernel_const_const_const(dimension_count_const, window_width_const, block_size_const) \
	convolution_tex_exact_blocked_kernel_kepler<dimension_count_const,window_width_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, packed_config_count);

#define launch_generic_kernel_const_const(dimension_count_const, block_size_const) \
	convolution_tex_generic_blocked_kernel_kepler<dimension_count_const,block_size_const><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(*output_buffer, input_tex, weights_tex, *data[1], packed_config_list, output_sizes, input_sizes, window_sizes, left_zero_padding, input_configuration_specific_striped.feature_map_count, output_configuration_specific.feature_map_count, entry_count, packed_config_count);

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

		template<int dimension_count>
		class convolution_layer_tester_cuda_kepler : public layer_tester_cuda
		{
		public:
			convolution_layer_tester_cuda_kepler()
			{
			}

			virtual ~convolution_layer_tester_cuda_kepler()
			{
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
				unsigned int entry_count)
			{
				cuda_util::copy_to_striped(
					*cuda_config,
					*input_buffers[0],
					*temporary_working_per_entry_buffer,
					input_elem_count_per_feature_map_list[0],
					input_configuration_specific_list[0].feature_map_count,
					entry_count,
					stream_id);

				cuda_texture weights_tex(data[0], 2);
				cuda_texture input_tex(temporary_working_per_entry_buffer, 2);

				const packed_config<dimension_count> * packed_config_list = static_cast<const packed_config<dimension_count> *>((const void *)*persistent_working_data[0]);

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					forward_x_block_count,
					packed_config_count,
					entry_count);

				launch_kernel(dimension_count, window_sizes[0], forward_x_block_size);
			}

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_data(layer_data::const_ptr host_data) const
			{
				std::vector<cuda_linear_buffer_device::const_ptr> res;

				if (host_data->size() != 2)
					return res;

				unsigned int window_total_size = 1;
				for(int i = 0; i < dimension_count; ++i)
					window_total_size *= window_sizes[i];
				unsigned int weight_count = output_configuration_specific.feature_map_count * input_configuration_specific_list[0].feature_map_count * window_total_size;
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
						bool second_feature_map_present = (input_feature_map_id_striped * 2 + 1 < input_configuration_specific_list[0].feature_map_count);
						for(int dst_elem_id = 0; dst_elem_id < static_cast<int>(window_total_size); ++dst_elem_id)
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
					cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*weights_striped.begin()), buffer_size, cudaMemcpyHostToDevice));
					res.push_back(new_buf);
				}

				{
					size_t buffer_size = host_data->at(1).size() * sizeof(float);
					cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*host_data->at(1).begin()), buffer_size, cudaMemcpyHostToDevice));
					res.push_back(new_buf);
				}

				return res;
			}

		protected:
			virtual void tester_configured()
			{
				nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

				input_configuration_specific_striped = cuda_util::get_layer_configuration_specific_striped(input_configuration_specific_list[0]);

				for(int i = 0; i < dimension_count; ++i)
				{
					window_sizes[i] = layer_derived->window_sizes[i];
					input_sizes[i] = input_configuration_specific_list[0].dimension_sizes[i];
					output_sizes[i] = output_configuration_specific.dimension_sizes[i];
					left_zero_padding[i] = layer_derived->left_zero_padding[i];
				}

				forward_x_block_size = get_block_size(output_sizes[0]);
				forward_x_block_count = (output_sizes[0] + forward_x_block_size - 1) / forward_x_block_size;
				forward_output_feature_map_block_count = (output_configuration_specific.feature_map_count + FEATURE_MAP_BLOCK_SIZE - 1) / FEATURE_MAP_BLOCK_SIZE;

				packed_config_count = forward_output_feature_map_block_count;
				for(int i = 1; i < dimension_count; ++i)
					packed_config_count *= output_sizes[i];
			}

			virtual size_t get_temporary_working_per_entry_buffer_size() const
			{
				return input_configuration_specific_striped.get_neuron_count() * sizeof(float2);
			}

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const
			{
				std::vector<unsigned int> res;
				res.push_back(input_configuration_specific_striped.get_neuron_count());
				return res;
			}

			std::vector<cuda_linear_buffer_device::const_ptr> get_persistent_working_data() const
			{
				std::vector<cuda_linear_buffer_device::const_ptr> res;

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

					res.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(&task_list[0], sizeof(packed_config<dimension_count>) * task_list.size())));
				}

				return res;
			}

		private:
			static int get_block_size(int output_width)
			{
				int block_count = (output_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
				int block_size = (output_width + block_count - 1) / block_count;
				return block_size;
			}

			array_by_val<int, dimension_count> output_sizes;
			array_by_val<int, dimension_count> input_sizes;
			array_by_val<int, dimension_count> window_sizes;
			array_by_val<int, dimension_count> left_zero_padding;

			layer_configuration_specific input_configuration_specific_striped;

			int forward_x_block_size;
			int forward_x_block_count;
			int forward_output_feature_map_block_count;
			int packed_config_count;
		};
	}
}
