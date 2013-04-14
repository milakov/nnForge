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

#include "layer_updater_cuda.h"

#include "util_cuda.h"

__global__ void dropout_kernel(
	float * __restrict neurons,
	const float * __restrict random_buf,
	float dropout_rate,
	int offset,
	unsigned int mask,
	int elem_count_per_feature_map,
	int feature_map_count,
	int entry_count)
{
	int internal_elem_id = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	bool in_bounds = (entry_id < entry_count) && (feature_map_id < feature_map_count) && (internal_elem_id < elem_count_per_feature_map);
	if (in_bounds)
	{
		unsigned int total_feature_map_id = entry_id * feature_map_count + feature_map_id;
		unsigned int random_elem_id = (total_feature_map_id + offset) & mask;
		if (random_buf[random_elem_id] < dropout_rate)
			neurons[total_feature_map_id * elem_count_per_feature_map + internal_elem_id] = 0.0F;
	}
}

namespace nnforge
{
	namespace cuda
	{
		layer_updater_cuda::layer_updater_cuda()
		{
		}

		layer_updater_cuda::~layer_updater_cuda()
		{
		}

		void layer_updater_cuda::configure(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			const_layer_smart_ptr layer_schema,
			cuda_running_configuration_const_smart_ptr cuda_config,
			bool backprop_required,
			bool different_input)
		{
			this->layer_schema = layer_schema;
			this->input_configuration_specific = input_configuration_specific;
			this->output_configuration_specific = output_configuration_specific;
			this->cuda_config = cuda_config;
			this->backprop_required = backprop_required;
			this->different_input = different_input;

			input_elem_count_per_entry = input_configuration_specific.get_neuron_count();
			output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			input_elem_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();

			updater_configured();
		}

		void layer_updater_cuda::updater_configured()
		{
		}

		std::vector<size_t> layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			return std::vector<size_t>();
		}

		std::vector<unsigned int> layer_updater_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}

		void layer_updater_cuda::update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const
		{
			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			buffer_configuration.add_per_entry_buffer(output_elem_count_per_entry * sizeof(float));

			if (backprop_required && !is_in_place_backprop())
				buffer_configuration.add_per_entry_buffer(input_elem_count_per_entry * sizeof(float));

			std::vector<unsigned int> tex_per_entry = get_linear_addressing_through_texture_per_entry();
			for(std::vector<unsigned int>::const_iterator it = tex_per_entry.begin(); it != tex_per_entry.end(); ++it)
				buffer_configuration.add_per_entry_linear_addressing_through_texture(*it);
		}

		void layer_updater_cuda::update_buffer_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
				buffer_configuration.add_constant_buffer(*it * updater_entry_count);

			buffer_configuration.add_constant_buffer(output_elem_count_per_entry * sizeof(float) * updater_entry_count);

			if (backprop_required && !is_in_place_backprop())
				buffer_configuration.add_constant_buffer(input_elem_count_per_entry * sizeof(float) * updater_entry_count);
		}

		layer_updater_cuda::buffer_set layer_updater_cuda::allocate_all_buffers(unsigned int max_entry_count) const
		{
			buffer_set res;

			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();

			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
			{
				// Allow safe float4 accesses
				size_t sz = *it * max_entry_count;
				res.additional_buffers.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz)));
			}

			{
				size_t sz = output_elem_count_per_entry * sizeof(float) * max_entry_count;
				res.output_neurons_buffer = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz));
			}

			if (backprop_required && !is_in_place_backprop())
			{
				size_t sz = input_elem_count_per_entry * sizeof(float) * max_entry_count;
				res.input_errors_buffer = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz));
			}

			return res;
		}

		void layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& training_speed,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
		}

		void layer_updater_cuda::enqueue_forward_dropout(
			cudaStream_t stream_id,
			const_cuda_linear_buffer_device_smart_ptr random_buffer,
			cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			float dropout_rate,
			unsigned int mask,
			unsigned int entry_count,
			unsigned int offset_in_random_list)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
			dropout_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*random_buffer,
				dropout_rate,
				offset_in_random_list,
				mask,
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
		}

		void layer_updater_cuda::enqueue_backward_dropout(
			cudaStream_t stream_id,
			const_cuda_linear_buffer_device_smart_ptr random_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			float dropout_rate,
			unsigned int mask,
			unsigned int entry_count,
			unsigned int offset_in_random_list)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
			dropout_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_errors_buffer,
				*random_buffer,
				dropout_rate,
				offset_in_random_list,
				mask,
				input_elem_count_per_feature_map,
				input_configuration_specific.feature_map_count,
				entry_count);
		}
	}
}
