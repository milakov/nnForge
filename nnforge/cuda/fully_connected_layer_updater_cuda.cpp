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

#include "fully_connected_layer_updater_cuda.h"

#include "cudnn_util.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cudnn_exception.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		fully_connected_layer_updater_cuda::fully_connected_layer_updater_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		fully_connected_layer_updater_cuda::~fully_connected_layer_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void fully_connected_layer_updater_cuda::enqueue_forward_propagation(
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
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				cublas_safe_call(cublasSgemm(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					output_elem_count_per_entry,
					entry_count,
					input_elem_count_per_entry_list[0],
					&alpha,
					*data[0],
					input_elem_count_per_entry_list[0],
					*input_buffers[0],
					input_elem_count_per_entry_list[0],
					&beta,
					*output_buffer,
					output_elem_count_per_entry));
			}

			if (bias)
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnAddTensor(
					cuda_config->get_cudnn_handle(),
					&alpha,
					bias_desc,
					*data[1],
					&beta,
					output_data_desc,
					*output_buffer));
			}
		}

		void fully_connected_layer_updater_cuda::enqueue_backward_data_propagation(
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
			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = (add_update_to_destination ? 1.0F : 0.0F);
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				input_elem_count_per_entry_list[0],
				entry_count,
				output_elem_count_per_entry,
				&alpha,
				*data[0],
				input_elem_count_per_entry_list[0],
				*output_errors_buffer,
				output_elem_count_per_entry,
				&beta,
				*input_errors_buffer,
				input_elem_count_per_entry_list[0]));
		}

		void fully_connected_layer_updater_cuda::enqueue_backward_weights_propagation(
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
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 1.0F;
				cublas_safe_call(cublasSgemm(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					input_elem_count_per_entry_list[0],
					output_elem_count_per_entry,
					entry_count,
					&alpha,
					*input_neurons_buffers[0],
					input_elem_count_per_entry_list[0],
					*output_errors_buffer,
					output_elem_count_per_entry,
					&beta,
					*gradient[0],
					input_elem_count_per_entry_list[0]));
			}

			// Update biases
			if (bias)
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
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

		void fully_connected_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);
			bias = layer_derived->bias;

			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));
		}

		int fully_connected_layer_updater_cuda::get_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}

		bool fully_connected_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool fully_connected_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool fully_connected_layer_updater_cuda::is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			return true;
		}
	}
}
