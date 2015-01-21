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

		void fully_connected_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
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
					input_elem_count_per_entry,
					&alpha,
					*data[0],
					input_elem_count_per_entry,
					(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
					input_elem_count_per_entry,
					&beta,
					*output_neurons_buffer,
					output_elem_count_per_entry));
			}

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
				cudnn_safe_call(cudnnAddTensor(
					cuda_config->get_cudnn_handle(),
					CUDNN_ADD_SAME_C,
					&alpha,
					bias_desc,
					*data[1],
					&beta,
					output_data_desc,
					*output_neurons_buffer));
			}
		}

		void fully_connected_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			if (!backprop_required)
				throw neural_network_exception("fully_connected_layer_updater_cuda is not configured to do backprop but requested to");

			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 0.0F;
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				input_elem_count_per_entry,
				entry_count,
				output_elem_count_per_entry,
				&alpha,
				*data[0],
				input_elem_count_per_entry,
				*output_errors_buffer,
				output_elem_count_per_entry,
				&beta,
				*input_errors_buffer,
				input_elem_count_per_entry));
		}

		void fully_connected_layer_updater_cuda::enqueue_update_weights(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
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
					input_elem_count_per_entry,
					output_elem_count_per_entry,
					entry_count,
					&alpha,
					(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
					input_elem_count_per_entry,
					*output_errors_buffer,
					output_elem_count_per_entry,
					&beta,
					*gradient[0],
					input_elem_count_per_entry));
			}

			// Update biases
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

		void fully_connected_layer_updater_cuda::updater_configured()
		{
			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				bias_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				1,
				output_configuration_specific.feature_map_count,
				1,
				1));
		}

		bool fully_connected_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}

		int fully_connected_layer_updater_cuda::get_block_size(int entry_count)
		{
			int block_size = std::min<int>(std::max<int>(static_cast<int>(sqrtf(static_cast<float>(entry_count))), 1), entry_count);
			return block_size;
		}
	}
}
