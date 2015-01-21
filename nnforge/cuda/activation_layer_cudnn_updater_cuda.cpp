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

#include "activation_layer_cudnn_updater_cuda.h"

#include "neural_network_cudnn_exception.h"
#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		activation_layer_cudnn_updater_cuda::activation_layer_cudnn_updater_cuda(cudnnActivationMode_t af)
			: af(af)
			, input_data_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_data_desc));
		}

		activation_layer_cudnn_updater_cuda::~activation_layer_cudnn_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(input_data_desc);
		}

		void activation_layer_cudnn_updater_cuda::enqueue_test(
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
			if (offset_input_entry_id > 0)
				throw neural_network_exception("activation_layer_cudnn_updater_cuda is not able to run using offset");

			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				input_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				input_configuration_specific.feature_map_count,
				1,
				input_elem_count_per_feature_map));

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnActivationForward(
				cuda_config->get_cudnn_handle(),
				af,
				&alpha,
				input_data_desc,
				*input_neurons_buffer,
				&beta,
				input_data_desc,
				*output_neurons_buffer));
		}

		void activation_layer_cudnn_updater_cuda::enqueue_backprop(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				input_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				input_configuration_specific.feature_map_count,
				1,
				input_elem_count_per_feature_map));

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnActivationBackward(
				cuda_config->get_cudnn_handle(),
				af,
				&alpha,
				input_data_desc,
				*output_neurons_buffer,
				input_data_desc,
				*output_errors_buffer,
				input_data_desc,
				*input_neurons_buffer,
				&beta,
				input_data_desc,
				*output_errors_buffer));
		}

		bool activation_layer_cudnn_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
