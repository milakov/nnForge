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

#include "reshape_layer_updater_cuda.h"

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "neural_network_cublas_exception.h"

namespace nnforge
{
	namespace cuda
	{
		reshape_layer_updater_cuda::reshape_layer_updater_cuda()
		{
		}

		reshape_layer_updater_cuda::~reshape_layer_updater_cuda()
		{
		}

		void reshape_layer_updater_cuda::enqueue_forward_propagation(
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
			if ((const float *)(*input_buffers[0]) != (const float *)(*output_buffer))
			{
				cuda_util::copy_buffer(
					*cuda_config,
					*input_buffers[0],
					*output_buffer,
					output_elem_count_per_entry * entry_count,
					stream_id);
			}
		}

		void reshape_layer_updater_cuda::enqueue_backward_data_propagation(
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
			unsigned int elem_count = entry_count * output_elem_count_per_entry;
			if (add_update_to_destination)
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				cublas_safe_call(cublasSaxpy(
					cuda_config->get_cublas_handle(),
					elem_count,
					&alpha,
					*output_errors_buffer,
					1,
					*input_errors_buffer,
					1));
			}
			else
			{
				if ((const float *)(*input_errors_buffer) != (const float *)(*output_errors_buffer))
				{
					cuda_util::copy_buffer(
						*cuda_config,
						*output_errors_buffer,
						*input_errors_buffer,
						output_elem_count_per_entry * entry_count,
						stream_id);
				}
			}
		}

		int reshape_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool reshape_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool reshape_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}
	}
}
