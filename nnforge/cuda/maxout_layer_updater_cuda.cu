/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "maxout_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../maxout_layer.h"
#include "../nn_types.h"

__global__ void maxout_upd_kernel(
	float * __restrict output,
	int * __restrict max_feature_map_positions,
	const float * __restrict input,
	int neuron_count_per_feature_map,
	int input_feature_map_count,
	int output_feature_map_count,
	int feature_map_subsampling_size,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	if ((neuron_id < neuron_count_per_feature_map) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int input_offset = (entry_id * input_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		float max_val = input[input_offset];
		int max_pos = 0;
		for(int i = 1; i < feature_map_subsampling_size; ++i)
		{
			input_offset += output_feature_map_count * neuron_count_per_feature_map;
			float new_val = input[input_offset];
			if (new_val > max_val)
			{
				max_val = new_val;
				max_pos = i;
			}
		}
		int output_offset = (entry_id * output_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		output[output_offset] = max_val;
		max_feature_map_positions[output_offset] = max_pos;
	}
}

template<bool add_update_to_destination>
__global__ void maxout_deriviative_upd_kernel(
	float * __restrict input_errors,
	const int * __restrict max_feature_map_positions,
	const float * __restrict output_errors,
	int neuron_count_per_feature_map,
	int input_feature_map_count,
	int output_feature_map_count,
	int feature_map_subsampling_size,
	int entry_count)
{
	int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
	int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

	if ((neuron_id < neuron_count_per_feature_map) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
	{
		int output_offset = (entry_id * output_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		int max_feature_map = max_feature_map_positions[output_offset];
		float output_error = output_errors[output_offset];

		int input_offset = (entry_id * input_feature_map_count + output_feature_map_id) * neuron_count_per_feature_map + neuron_id;
		for(int i = 0; i < feature_map_subsampling_size; ++i)
		{
			if (add_update_to_destination)
				input_errors[input_offset] += ((i == max_feature_map) ? output_error : 0.0F);
			else
				input_errors[input_offset] = ((i == max_feature_map) ? output_error : 0.0F);
			input_offset += output_feature_map_count * neuron_count_per_feature_map;
		}
	}
}

namespace nnforge
{
	namespace cuda
	{
		maxout_layer_updater_cuda::maxout_layer_updater_cuda()
		{
		}

		maxout_layer_updater_cuda::~maxout_layer_updater_cuda()
		{
		}

		void maxout_layer_updater_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);

			maxout_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*temporary_per_entry_buffer,
				*input_buffers[0],
				output_elem_count_per_feature_map,
				input_configuration_specific_list[0].feature_map_count,
				output_configuration_specific.feature_map_count,
				feature_map_subsampling_size,
				entry_count);
		}

		void maxout_layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);

			if (add_update_to_destination)
				maxout_deriviative_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*temporary_per_entry_buffer,
					*output_errors_buffer,
					output_elem_count_per_feature_map,
					input_configuration_specific_list[0].feature_map_count,
					output_configuration_specific.feature_map_count,
					feature_map_subsampling_size,
					entry_count);
			else
				maxout_deriviative_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_errors_buffer,
					*temporary_per_entry_buffer,
					*output_errors_buffer,
					output_elem_count_per_feature_map,
					input_configuration_specific_list[0].feature_map_count,
					output_configuration_specific.feature_map_count,
					feature_map_subsampling_size,
					entry_count);
		}

		bool maxout_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool maxout_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool maxout_layer_updater_cuda::is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const
		{
			return true;
		}

		void maxout_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const maxout_layer> layer_derived = nnforge_dynamic_pointer_cast<const maxout_layer>(layer_schema);

			feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
		}

		size_t maxout_layer_updater_cuda::get_temporary_per_entry_buffer_size() const
		{
			return output_elem_count_per_entry * sizeof(int);
		}
	}
}
