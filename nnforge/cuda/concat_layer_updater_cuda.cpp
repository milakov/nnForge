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

#include "concat_layer_updater_cuda.h"

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"
#include "neural_network_cublas_exception.h"

namespace nnforge
{
	namespace cuda
	{
		concat_layer_updater_cuda::concat_layer_updater_cuda()
		{
		}

		concat_layer_updater_cuda::~concat_layer_updater_cuda()
		{
		}

		void concat_layer_updater_cuda::enqueue_forward_propagation(
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
			unsigned int offset = 0;
			for(unsigned int i = 0; i < static_cast<unsigned int>(input_configuration_specific_list.size()); ++i)
			{
				unsigned int elem_count = input_elem_count_per_entry_list[i] * entry_count;

				if ((offset & 3) == 0)
					cuda_util::copy_buffer(
						*cuda_config,
						*input_buffers[i],
						(float *)(*output_buffer) + offset,
						elem_count,
						stream_id);
				else
					cuda_safe_call(cudaMemcpyAsync(
						(float *)(*output_buffer) + offset,
						*input_buffers[i],
						elem_count * sizeof(float),
						cudaMemcpyDeviceToDevice,
						stream_id));

				offset += elem_count;
			}
		}

		void concat_layer_updater_cuda::enqueue_backward_data_propagation(
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
			unsigned int offset = 0;
			for(unsigned int i = 0; i < input_index; ++i)
			{
				unsigned int elem_count = input_elem_count_per_entry_list[i] * entry_count;
				offset += elem_count;
			}
			unsigned int elem_count = input_elem_count_per_entry_list[input_index] * entry_count;

			if (add_update_to_destination)
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				cublas_safe_call(cublasSaxpy(
					cuda_config->get_cublas_handle(),
					elem_count,
					&alpha,
					(const float *)(*output_errors_buffer) + offset,
					1,
					*input_errors_buffer,
					1));
			}
			else
			{
				if ((offset & 3) == 0)
					cuda_util::copy_buffer(
						*cuda_config,
						(const float *)(*output_errors_buffer) + offset,
						*input_errors_buffer,
						elem_count,
						stream_id);
				else
					cuda_safe_call(cudaMemcpyAsync(
						*input_errors_buffer,
						(const float *)(*output_errors_buffer) + offset,
						elem_count * sizeof(float),
						cudaMemcpyDeviceToDevice,
						stream_id));
			}
		}

		bool concat_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool concat_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}
	}
}
