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

#include "softmax_layer_tester_cuda.h"

#include "cudnn_util.h"
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

		void softmax_layer_tester_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			unsigned int entry_count)
		{
			cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));

			cudnn_util::set_tensor_descriptor(
				input_data_desc,
				output_configuration_specific,
				entry_count);

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnSoftmaxForward(
				cuda_config->get_cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				input_data_desc,
				*input_buffers[0],
				&beta,
				input_data_desc,
				*output_buffer));
		}
	}
}
