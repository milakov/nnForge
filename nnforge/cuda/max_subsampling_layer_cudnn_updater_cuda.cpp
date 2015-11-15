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

#include "max_subsampling_layer_cudnn_updater_cuda.h"

#include "neural_network_cudnn_exception.h"
#include "cudnn_util.h"
#include "../max_subsampling_layer.h"

namespace nnforge
{
	namespace cuda
	{
		max_subsampling_layer_cudnn_updater_cuda::max_subsampling_layer_cudnn_updater_cuda()
			: input_data_desc(0)
			, output_data_desc(0)
			, subsampling_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreatePoolingDescriptor(&subsampling_desc));
		}

		max_subsampling_layer_cudnn_updater_cuda::~max_subsampling_layer_cudnn_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(input_data_desc);
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyPoolingDescriptor(subsampling_desc);
		}

		void max_subsampling_layer_cudnn_updater_cuda::enqueue_forward_propagation(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				input_configuration_specific_list[0],
				entry_count);
			cudnn_util::set_tensor_descriptor(
				output_data_desc,
				output_configuration_specific,
				entry_count);

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnPoolingForward(
				cuda_config->get_cudnn_handle(),
				subsampling_desc,
				&alpha,
				input_data_desc,
				*input_buffers[0],
				&beta,
				output_data_desc,
				*output_buffer));
		}

		void max_subsampling_layer_cudnn_updater_cuda::enqueue_backward_data_propagation(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				input_configuration_specific_list[0],
				entry_count);
			cudnn_util::set_tensor_descriptor(
				output_data_desc,
				output_configuration_specific,
				entry_count);

			float alpha = 1.0F;
			float beta = (add_update_to_destination ? 1.0F : 0.0F);
			cudnn_safe_call(cudnnPoolingBackward(
				cuda_config->get_cudnn_handle(),
				subsampling_desc,
				&alpha,
				output_data_desc,
				*output_neurons_buffer,
				output_data_desc,
				*output_errors_buffer,
				input_data_desc,
				*input_neurons_buffers[0],
				&beta,
				input_data_desc,
				*input_errors_buffer));
		}

		void max_subsampling_layer_cudnn_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const max_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const max_subsampling_layer>(layer_schema);

			cudnn_util::set_pooling_descriptor(
				subsampling_desc,
				CUDNN_POOLING_MAX,
				layer_derived->subsampling_sizes);
		}

		bool max_subsampling_layer_cudnn_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return true;
		}

		bool max_subsampling_layer_cudnn_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return true;
		}
	}
}
