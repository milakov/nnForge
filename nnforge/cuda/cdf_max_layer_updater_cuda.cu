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

#include "cdf_max_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../cdf_max_layer.h"
#include <memory>

namespace nnforge
{
	namespace cuda
	{
		template<bool IS_MIN>
		__global__ void cdf_max_upd_kernel(
			float * __restrict output,
			const float * __restrict input,
			int neuron_count,
			int entry_subsampling_size,
			int output_entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			if ((neuron_id < neuron_count) && (output_entry_id < output_entry_count))
			{
				int input_offset = output_entry_id * neuron_count * entry_subsampling_size + neuron_id;
				float product = 1.0F;
				#pragma unroll 4
				for(int i = 0; i < entry_subsampling_size; ++i)
				{
					float val = input[input_offset];
					if (IS_MIN)
						product *= (1.0F - val);
					else
						product *= val;
					input_offset += neuron_count;
				}
				if (IS_MIN)
					product = 1.0F - product;
				output[output_entry_id * neuron_count + neuron_id] = product;
			}
		}

		template<bool add_update_to_destination,bool IS_MIN>
		__global__ void cdf_max_backprop_upd_kernel(
			float * __restrict input_errors,
			const float * __restrict output_errors,
			const float * __restrict input_neurons,
			const float * __restrict output_neurons,
			int neuron_count,
			int entry_subsampling_size,
			int output_entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_entry_id = blockIdx.y * blockDim.y + threadIdx.y;

			if ((neuron_id < neuron_count) && (output_entry_id < output_entry_count))
			{
				int output_offset = output_entry_id * neuron_count + neuron_id;
				float output_error = output_errors[output_offset];
				float output_neuron = output_neurons[output_offset];
				float mult = output_error * (IS_MIN ? 1.0F - output_neuron : output_neuron);

				int input_offset = output_entry_id * neuron_count * entry_subsampling_size + neuron_id;
				#pragma unroll 4
				for(unsigned int i = 0; i < entry_subsampling_size; ++i)
				{
					float err = 0.0F;
					if (mult != 0.0F)
					{
						float input_val = input_neurons[input_offset];
						err = __fdividef(mult, (IS_MIN ? 1.0F - input_val : input_val));
					}
					if (add_update_to_destination)
						input_errors[input_offset] += err;
					else
						input_errors[input_offset] = err;

					input_offset += neuron_count;
				}
			}
		}

		void cdf_max_layer_updater_cuda::enqueue_forward_propagation(
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
				output_elem_count_per_entry,
				entry_count,
				1);

			if (is_min)
				cdf_max_upd_kernel<true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_elem_count_per_entry,
					entry_subsampling_size,
					entry_count);
			else
				cdf_max_upd_kernel<false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_buffer,
					*input_buffers[0],
					output_elem_count_per_entry,
					entry_subsampling_size,
					entry_count);
		}

		void cdf_max_layer_updater_cuda::enqueue_backward_data_propagation(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_entry,
				entry_count,
				1);

			if (add_update_to_destination)
			{
				if (is_min)
					cdf_max_backprop_upd_kernel<true,true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						*input_neurons_buffers[0],
						*output_neurons_buffer,
						output_elem_count_per_entry,
						entry_subsampling_size,
						entry_count);
				else
					cdf_max_backprop_upd_kernel<true,false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						*input_neurons_buffers[0],
						*output_neurons_buffer,
						output_elem_count_per_entry,
						entry_subsampling_size,
						entry_count);
			}
			else
			{
				if (is_min)
					cdf_max_backprop_upd_kernel<false,true><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						*input_neurons_buffers[0],
						*output_neurons_buffer,
						output_elem_count_per_entry,
						entry_subsampling_size,
						entry_count);
				else
					cdf_max_backprop_upd_kernel<false,false><<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
						*input_errors_buffer,
						*output_errors_buffer,
						*input_neurons_buffers[0],
						*output_neurons_buffer,
						output_elem_count_per_entry,
						entry_subsampling_size,
						entry_count);
			}
		}

		bool cdf_max_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return true;
		}

		bool cdf_max_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return true;
		}

		void cdf_max_layer_updater_cuda::updater_configured()
		{
			std::shared_ptr<const cdf_max_layer> layer_derived = std::dynamic_pointer_cast<const cdf_max_layer>(layer_schema);

			entry_subsampling_size = layer_derived->entry_subsampling_size;
			is_min = layer_derived->is_min;
		}
	}
}
