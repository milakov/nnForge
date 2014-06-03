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

#include "convolution_weight_vector_bound_cuda.h"

#include "../convolution_layer.h"
#include "../nn_types.h"

extern __shared__ float arr[];
template<bool single_item_per_thread>
__global__ void convolution_normalize_weights_to_max_l2_norm_kernel(
	float * __restrict weights,
	const float * __restrict weights_read_copy,
	float max_l2_norm_squared,
	int incoming_weight_count_per_output_neuron,
	int output_feature_map_count,
	int min_iteration_count)
{
	int thread_id = threadIdx.x;
	int output_feature_map_id = blockIdx.y;
	int entry_id = blockIdx.z;
	int threadblock_size = blockDim.x;

	int base_weight_id = (entry_id * output_feature_map_count + output_feature_map_id) * incoming_weight_count_per_output_neuron;
	float current_val;
	float sum = 0.0F;
	int current_weight_id = thread_id;
	for(int i = 0; i < min_iteration_count; ++i)
	{
		current_val = weights_read_copy[base_weight_id + current_weight_id];
		sum += current_val * current_val;
		current_weight_id += threadblock_size;
	}
	if (current_weight_id < incoming_weight_count_per_output_neuron)
	{
		current_val = weights_read_copy[base_weight_id + current_weight_id];
		sum += current_val * current_val;
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

	sum = arr[0];
	if (sum <= max_l2_norm_squared)
		return;

	float mult = rsqrtf(__fdividef(sum, max_l2_norm_squared));

	if (single_item_per_thread)
	{
		if (thread_id < incoming_weight_count_per_output_neuron)
			weights[base_weight_id + thread_id] = current_val * mult;
	}
	else
	{
		int current_weight_id = thread_id;
		for(int i = 0; i < min_iteration_count; ++i)
		{
			weights[base_weight_id + current_weight_id] = weights_read_copy[base_weight_id + current_weight_id] * mult;
			current_weight_id += threadblock_size;
		}
		if (current_weight_id < incoming_weight_count_per_output_neuron)
			weights[base_weight_id + current_weight_id] = weights_read_copy[base_weight_id + current_weight_id] * mult;
	}
}

namespace nnforge
{
	namespace cuda
	{
		convolution_weight_vector_bound_cuda::convolution_weight_vector_bound_cuda()
		{
		}

		convolution_weight_vector_bound_cuda::~convolution_weight_vector_bound_cuda()
		{
		}

		const boost::uuids::uuid& convolution_weight_vector_bound_cuda::get_uuid() const
		{
			return convolution_layer::layer_guid;
		}

		void convolution_weight_vector_bound_cuda::enqueue_normalize_weights(
			cudaStream_t stream_id,
			const weight_vector_bound& bound,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count,
			const std::vector<unsigned int>& incoming_weight_count_per_output_neuron_list)
		{
			int threadblock_size = get_threadblock_size(incoming_weight_count_per_output_neuron_list[0]);
			dim3 grid_size(1, output_feature_map_count, entry_count);
			dim3 block_size(threadblock_size, 1, 1);
			int min_iteration_count = incoming_weight_count_per_output_neuron_list[0] / threadblock_size;
			int smem_size = threadblock_size * sizeof(float);
			float max_l2_norm_squared = bound.max_l2_norm * bound.max_l2_norm;

			if (incoming_weight_count_per_output_neuron_list[0] <= threadblock_size)
			{
				convolution_normalize_weights_to_max_l2_norm_kernel<true><<<grid_size, block_size, smem_size, stream_id>>>(
					*data[0],
					*data[0],
					max_l2_norm_squared,
					incoming_weight_count_per_output_neuron_list[0],
					output_feature_map_count,
					min_iteration_count);
			}
			else
			{
				convolution_normalize_weights_to_max_l2_norm_kernel<false><<<grid_size, block_size, smem_size, stream_id>>>(
					*data[0],
					*data[0],
					max_l2_norm_squared,
					incoming_weight_count_per_output_neuron_list[0],
					output_feature_map_count,
					min_iteration_count);
			}
		}

		weight_vector_bound_cuda_smart_ptr convolution_weight_vector_bound_cuda::create_specific() const
		{
			return weight_vector_bound_cuda_smart_ptr(new convolution_weight_vector_bound_cuda());
		}

		void convolution_weight_vector_bound_cuda::weight_vector_bound_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			output_feature_map_count = layer_derived->output_feature_map_count;
		}

		int convolution_weight_vector_bound_cuda::get_threadblock_size(int incoming_weight_count_per_output_neuron)
		{
			int threadblock_size;

			if (incoming_weight_count_per_output_neuron < 256)
			{
				threadblock_size = (incoming_weight_count_per_output_neuron + 32 - 1) / 32 * 32;
			}
			else
			{
				int threadblock_count = (incoming_weight_count_per_output_neuron + 256 - 1) / 256;
				threadblock_size = (incoming_weight_count_per_output_neuron + threadblock_count - 1) / threadblock_count;
				threadblock_size = (threadblock_size + 32 - 1) / 32 * 32;
			}

			return threadblock_size;
		}
	}
}
