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

#include "softmax_layer_tester_cuda.h"

#include "neural_network_cudnn_exception.h"

namespace nnforge
{
	namespace cuda
	{
		softmax_layer_tester_cuda::softmax_layer_tester_cuda()
			: input_data_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_data_desc));
		}

		softmax_layer_tester_cuda::~softmax_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(input_data_desc);
		}

		void softmax_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
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

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnSoftmaxForward(
				cuda_config->get_cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				input_data_desc,
				*input_buffer,
				&beta,
				input_data_desc,
				*additional_buffers[0]));
		}

		cuda_linear_buffer_device_smart_ptr softmax_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		std::vector<size_t> softmax_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}
	}
}
