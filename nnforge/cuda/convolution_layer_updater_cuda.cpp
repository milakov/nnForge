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

#include "convolution_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "cudnn_util.h"
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

		void convolution_layer_updater_cuda::enqueue_forward_propagation(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				input_configuration_specific_list[0],
				entry_count);
			cudnn_util::set_tensor_descriptor(
				output_data_desc,
				output_configuration_specific,
				entry_count);

			{
				void * workspace = 0;
				size_t workspace_size = 0;
				if (temporary_working_fixed_buffer)
				{
					workspace = *temporary_working_fixed_buffer;
					workspace_size = temporary_working_fixed_buffer->get_size();
				}

				cudnnConvolutionFwdAlgo_t algo = cuda_config->cudnn_find_convolution_forward_algo(
					input_data_desc,
					weights_desc,
					convolution_desc,
					output_data_desc,
					*input_buffers[0],
					*data[0],
					*output_buffer,
					workspace,
					workspace_size);

				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnConvolutionForward(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_data_desc,
					*input_buffers[0],
					weights_desc,
					*data[0],
					convolution_desc,
					algo,
					workspace,
					workspace_size,
					&beta,
					output_data_desc,
					*output_buffer));
			}

			if (bias)
			{
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

		void convolution_layer_updater_cuda::enqueue_backward_data_propagation(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				input_configuration_specific_list[0],
				entry_count);
			cudnn_util::set_tensor_descriptor(
				output_data_desc,
				output_configuration_specific,
				entry_count);

			{
				void * workspace = 0;
				size_t workspace_size = 0;
				if (temporary_working_fixed_buffer)
				{
					workspace = *temporary_working_fixed_buffer;
					workspace_size = temporary_working_fixed_buffer->get_size();
				}

				cudnnConvolutionBwdDataAlgo_t algo = cuda_config->cudnn_find_convolution_backward_data_algo(
					input_data_desc,
					weights_desc,
					convolution_desc,
					output_data_desc,
					*output_errors_buffer,
					*data[0],
					*temporary_working_per_entry_buffer,
					workspace,
					workspace_size);

				float alpha = 1.0F;
				float beta = (add_update_to_destination ? 1.0F : 0.0F);
				cudnn_safe_call(cudnnConvolutionBackwardData(
					cuda_config->get_cudnn_handle(),
					&alpha,
					weights_desc,
					*data[0],
					output_data_desc,
					*output_errors_buffer,
					convolution_desc,
					algo,
					workspace,
					workspace_size,
					&beta,
					input_data_desc,
					*input_errors_buffer));
			}
		}

		void convolution_layer_updater_cuda::enqueue_backward_weights_propagation(
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
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				input_configuration_specific_list[0],
				entry_count);
			cudnn_util::set_tensor_descriptor(
				output_data_desc,
				output_configuration_specific,
				entry_count);

			{
				void * workspace = 0;
				size_t workspace_size = 0;
				if (temporary_working_fixed_buffer)
				{
					workspace = *temporary_working_fixed_buffer;
					workspace_size = temporary_working_fixed_buffer->get_size();
				}

				cudnnConvolutionBwdFilterAlgo_t algo = cuda_config->cudnn_find_convolution_backward_weights_algo(
					input_data_desc,
					weights_desc,
					convolution_desc,
					output_data_desc,
					*input_neurons_buffers[0],
					*output_errors_buffer,
					(unsigned char *)workspace,
					(unsigned char *)workspace + update_weights_find_algo_working_buffer_size,
					workspace_size - update_weights_find_algo_working_buffer_size);

				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnConvolutionBackwardFilter(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_data_desc,
					*input_neurons_buffers[0],
					output_data_desc,
					*output_errors_buffer,
					convolution_desc,
					algo,
					workspace,
					workspace_size,
					&beta,
					weights_desc,
					*gradient[0]));
			}

			if (bias)
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
			std::shared_ptr<const convolution_layer> layer_derived = std::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			window_sizes = layer_derived->window_sizes;
			strides = layer_derived->strides;
			bias = layer_derived->bias;

			std::vector<unsigned int> zero_padding = layer_derived->left_zero_padding;
			for(int i = 0; i < window_sizes.size(); ++i)
			{
				if (zero_padding[i] != layer_derived->right_zero_padding[i])
					throw neural_network_exception("cuDNN is not able to run convolution when left and right padding sizes don't match");
			}

			cudnn_util::set_filter_descriptor(
				weights_desc,
				output_configuration_specific.feature_map_count,
				input_configuration_specific_list[0].feature_map_count,
				window_sizes);

			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));

			cudnn_util::set_convolution_descriptor(
				convolution_desc,
				zero_padding,
				strides);

			unsigned int update_weights_find_algo_working_buffer_elem_count = input_configuration_specific_list[0].feature_map_count * output_configuration_specific.feature_map_count;
			for(int i = 0; i < window_sizes.size(); ++i)
				update_weights_find_algo_working_buffer_elem_count *= window_sizes[i];
			update_weights_find_algo_working_buffer_size = update_weights_find_algo_working_buffer_elem_count * sizeof(float);
			update_weights_find_algo_working_buffer_size = (update_weights_find_algo_working_buffer_size + 16 - 1) / 16 * 16;
			unsigned int update_weights_working_buffer_elem_count = std::max(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
			for(int i = 0; i < window_sizes.size(); ++i)
				update_weights_working_buffer_elem_count *= window_sizes[i];
			update_weights_working_buffer_size = update_weights_working_buffer_elem_count * sizeof(int);
			update_weights_working_buffer_size = std::max(update_weights_working_buffer_size, (size_t)(1024*1024));
		}

		std::pair<size_t, bool> convolution_layer_updater_cuda::get_temporary_working_fixed_buffer_size(const layer_action& action) const
		{
			bool is_over_sol_algos_available = cudnn_util::is_over_sol_algos_available(window_sizes, strides);
			switch (action.get_action_type())
			{
			case layer_action::forward:
			case layer_action::backward_data:
				{
					unsigned int working_buffer_elem_count = std::max(input_configuration_specific_list[0].feature_map_count, output_configuration_specific.feature_map_count);
					for(int i = 0; i < window_sizes.size(); ++i)
						working_buffer_elem_count *= window_sizes[i];
					return std::make_pair(std::max(working_buffer_elem_count * sizeof(int), (size_t)(1024*1024)), is_over_sol_algos_available);
				}
			case layer_action::backward_weights:
				{
					return std::make_pair(update_weights_find_algo_working_buffer_size + update_weights_working_buffer_size, is_over_sol_algos_available);
				}
			default:
				return std::make_pair(0, false);
			}
		}

		size_t convolution_layer_updater_cuda::get_temporary_working_per_entry_buffer_size(const layer_action& action) const
		{
			if (action.get_action_type() == layer_action::backward_data)
			{
				return input_elem_count_per_entry_list[0] * sizeof(float);
			}
			else
				return 0;
		}

		bool convolution_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool convolution_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}

		bool convolution_layer_updater_cuda::is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			return true;
		}
	}
}
