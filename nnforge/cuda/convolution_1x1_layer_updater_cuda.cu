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

#include "convolution_1x1_layer_updater_cuda.h"

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cuda_exception.h"
#include "neural_network_cudnn_exception.h"

#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		convolution_1x1_layer_updater_cuda::convolution_1x1_layer_updater_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		convolution_1x1_layer_updater_cuda::~convolution_1x1_layer_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void convolution_1x1_layer_updater_cuda::enqueue_test(
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
				cuda_util::transpose(
					*cuda_config,
					(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
					*additional_buffers[0],
					input_elem_count_per_feature_map,
					input_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				cublas_safe_call(cublasSgemm(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					output_configuration_specific.feature_map_count,
					entry_count * input_elem_count_per_feature_map,
					input_configuration_specific.feature_map_count,
					&alpha,
					*data[0],
					input_configuration_specific.feature_map_count,
					*additional_buffers[0],
					input_configuration_specific.feature_map_count,
					&beta,
					*additional_buffers[1],
					output_configuration_specific.feature_map_count));

				cuda_util::transpose(
					*cuda_config,
					*additional_buffers[1],
					*output_neurons_buffer,
					output_configuration_specific.feature_map_count,
					output_elem_count_per_feature_map,
					entry_count,
					stream_id);
			}

			// Add bias
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

		void convolution_1x1_layer_updater_cuda::enqueue_backprop(
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
			cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 0.0F;
			cublas_safe_call(cublasSgemm(
				cuda_config->get_cublas_handle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				input_configuration_specific.feature_map_count,
				entry_count * input_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				&alpha,
				*data[0],
				input_configuration_specific.feature_map_count,
				*additional_buffers[1],
				output_configuration_specific.feature_map_count,
				&beta,
				*additional_buffers[0],
				input_configuration_specific.feature_map_count));

			cuda_util::transpose(
				*cuda_config,
				*additional_buffers[0],
				*input_errors_buffer,
				input_configuration_specific.feature_map_count,
				input_elem_count_per_feature_map,
				entry_count,
				stream_id);
		}

		void convolution_1x1_layer_updater_cuda::enqueue_update_weights(
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
				cuda_util::transpose(
					*cuda_config,
					*output_errors_buffer,
					*additional_buffers[1],
					output_elem_count_per_feature_map,
					output_configuration_specific.feature_map_count,
					entry_count,
					stream_id);

				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 1.0F;
				cublas_safe_call(cublasSgemm(
					cuda_config->get_cublas_handle(),
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					input_configuration_specific.feature_map_count,
					output_configuration_specific.feature_map_count,
					entry_count * input_elem_count_per_feature_map,
					&alpha,
					*additional_buffers[0],
					input_configuration_specific.feature_map_count,
					*additional_buffers[1],
					output_configuration_specific.feature_map_count,
					&beta,
					*gradient[0],
					input_configuration_specific.feature_map_count));
			}

			// Update bias
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

		void convolution_1x1_layer_updater_cuda::updater_configured()
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

		std::vector<size_t> convolution_1x1_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(input_elem_count_per_entry * sizeof(float));
			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		bool convolution_1x1_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}
	}
}
