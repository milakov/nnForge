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

#include "convolution_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cudnn_exception.h"
#include "neural_network_cuda_exception.h"
#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		convolution_layer_updater_cuda::convolution_layer_updater_cuda()
			: input_data_desc(0)
			, output_data_desc(0)
			, weights_desc(0)
			, convolution_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateFilterDescriptor(&weights_desc));
			cudnn_safe_call(cudnnCreateConvolutionDescriptor(&convolution_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		convolution_layer_updater_cuda::~convolution_layer_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(input_data_desc);
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyFilterDescriptor(weights_desc);
			cudnnDestroyConvolutionDescriptor(convolution_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void convolution_layer_updater_cuda::enqueue_test(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				input_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				input_configuration_specific.feature_map_count,
				(input_configuration_specific.dimension_sizes.size() > 1) ? input_configuration_specific.dimension_sizes[1] : 1,
				input_configuration_specific.dimension_sizes[0]));
			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				output_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				output_configuration_specific.feature_map_count,
				(output_configuration_specific.dimension_sizes.size() > 1) ? output_configuration_specific.dimension_sizes[1] : 1,
				output_configuration_specific.dimension_sizes[0]));

			{
				cudnnConvolutionFwdAlgo_t algo;
				cudnn_safe_call(cudnnGetConvolutionForwardAlgorithm(
					cuda_config->get_cudnn_handle(),
					input_data_desc,
					weights_desc,
					convolution_desc,
					output_data_desc,
					CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
					additional_buffers[0]->get_size(),
					&algo));

				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnConvolutionForward(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_data_desc,
					(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
					weights_desc,
					*data[0],
					convolution_desc,
					algo,
					*additional_buffers[0],
					additional_buffers[0]->get_size(),
					&beta,
					output_data_desc,
					*output_neurons_buffer));
			}

			{
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

		void convolution_layer_updater_cuda::enqueue_backprop(
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
				throw neural_network_exception("convolution_layer_updater_cuda is not configured to do backprop but requested to");

			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				input_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				input_configuration_specific.feature_map_count,
				(input_configuration_specific.dimension_sizes.size() > 1) ? input_configuration_specific.dimension_sizes[1] : 1,
				input_configuration_specific.dimension_sizes[0]));
			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				output_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				output_configuration_specific.feature_map_count,
				(output_configuration_specific.dimension_sizes.size() > 1) ? output_configuration_specific.dimension_sizes[1] : 1,
				output_configuration_specific.dimension_sizes[0]));

			{
				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnConvolutionBackwardData(
					cuda_config->get_cudnn_handle(),
					&alpha,
					weights_desc,
					*data[0],
					output_data_desc,
					*output_errors_buffer,
					convolution_desc,
					&beta,
					input_data_desc,
					*input_errors_buffer));
			}
		}

		void convolution_layer_updater_cuda::enqueue_update_weights(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				input_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				input_configuration_specific.feature_map_count,
				(input_configuration_specific.dimension_sizes.size() > 1) ? input_configuration_specific.dimension_sizes[1] : 1,
				input_configuration_specific.dimension_sizes[0]));
			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				output_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				output_configuration_specific.feature_map_count,
				(output_configuration_specific.dimension_sizes.size() > 1) ? output_configuration_specific.dimension_sizes[1] : 1,
				output_configuration_specific.dimension_sizes[0]));

			{
				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnConvolutionBackwardFilter(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_data_desc,
					(const float *)(*input_neurons_buffer) + input_elem_count_per_entry * offset_input_entry_id,
					output_data_desc,
					*output_errors_buffer,
					convolution_desc,
					&beta,
					weights_desc,
					*gradient[0]));
			}

			{
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

		void convolution_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			window_sizes = layer_derived->window_sizes;

			zero_padding = layer_derived->left_zero_padding;
			for(int i = 0; i < window_sizes.size(); ++i)
			{
				if (zero_padding[i] != layer_derived->right_zero_padding[i])
					throw neural_network_exception("cuDNN is not able to run convolution when left and right padding sizes don't match");
			}

			cudnn_safe_call(cudnnSetFilter4dDescriptor(
				weights_desc,
				CUDNN_DATA_FLOAT,
				output_configuration_specific.feature_map_count,
				input_configuration_specific.feature_map_count,
				(window_sizes.size() > 1) ? window_sizes[1] : 1,
				window_sizes[0]));

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				bias_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				1,
				output_configuration_specific.feature_map_count,
				1,
				1));

			cudnn_safe_call(cudnnSetConvolution2dDescriptor(
				convolution_desc,
				(zero_padding.size() > 1) ? zero_padding[1] : 1,
				zero_padding[0],
				1,
				1,
				1,
				1,
				CUDNN_CROSS_CORRELATION));
		}

		std::vector<size_t> convolution_layer_updater_cuda::get_sizes_of_additional_buffers_fixed() const
		{
			std::vector<size_t> res;

			unsigned int working_buffer_elem_count = input_configuration_specific.feature_map_count;
			for(int i = 0; i < window_sizes.size(); ++i)
				working_buffer_elem_count *= window_sizes[i];
			res.push_back(working_buffer_elem_count * sizeof(int));

			return res;
		}

		bool convolution_layer_updater_cuda::is_in_place_backprop() const
		{
			return false;
		}
	}
}
