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
#include "cudnn_util.h"
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

		void fully_connected_layer_tester_cuda::enqueue_forward_propagation(
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
				else
				{
					cublasSgemv(
						cuda_config->get_cublas_handle(),
						CUBLAS_OP_T,
						input_elem_count_per_entry_list[0],
						output_elem_count_per_entry,
						&alpha,
						*data[0],
						input_elem_count_per_entry_list[0],
						*input_buffers[0],
						1,
						&beta,
						*output_buffer,
						1);
				}
			}

			// Add bias
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
				float alpha = 1.0F;
				float beta = 1.0F;
				cudnn_safe_call(cudnnAddTensor_v3(
					cuda_config->get_cudnn_handle(),
					&alpha,
					bias_desc,
					*data[1],
					&beta,
					output_data_desc,
					*output_buffer));
			}
		}

		void fully_connected_layer_tester_cuda::tester_configured()
		{
			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));
		}
	}
}
