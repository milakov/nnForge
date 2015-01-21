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

#include "fully_connected_layer_tester_cuda.h"

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cudnn_exception.h"

namespace nnforge
{
	namespace cuda
	{
		fully_connected_layer_tester_cuda::fully_connected_layer_tester_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		fully_connected_layer_tester_cuda::~fully_connected_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void fully_connected_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				if (entry_count > 1)
				{
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
						*input_buffer,
						input_elem_count_per_entry,
						&beta,
						*additional_buffers[0],
						output_elem_count_per_entry));
				}
				else
				{
					cublasSgemv(
						cuda_config->get_cublas_handle(),
						CUBLAS_OP_T,
						input_elem_count_per_entry,
						output_elem_count_per_entry,
						&alpha,
						*data[0],
						input_elem_count_per_entry,
						*input_buffer,
						1,
						&beta,
						*additional_buffers[0],
						1);
				}
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
					1));
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
					*additional_buffers[0]));
			}
		}

		void fully_connected_layer_tester_cuda::tester_configured()
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

		std::vector<size_t> fully_connected_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr fully_connected_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}
	}
}
