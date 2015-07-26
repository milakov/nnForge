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

#include "convolution_1x1_layer_tester_cuda.h"

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cudnn_exception.h"

#include "../convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		convolution_1x1_layer_tester_cuda::convolution_1x1_layer_tester_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		convolution_1x1_layer_tester_cuda::~convolution_1x1_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void convolution_1x1_layer_tester_cuda::enqueue_forward_propagation(
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
				float * temporary_working_buffer1 = *temporary_working_per_entry_buffer;
				float * temporary_working_buffer2 = temporary_working_buffer1 + entry_count * input_elem_count_per_entry_list[0];

				cuda_util::transpose(
					*cuda_config,
					*input_buffers[0],
					temporary_working_buffer1,
					input_elem_count_per_feature_map_list[0],
					input_configuration_specific_list[0].feature_map_count,
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
					entry_count * input_elem_count_per_feature_map_list[0],
					input_configuration_specific_list[0].feature_map_count,
					&alpha,
					*data[0],
					input_configuration_specific_list[0].feature_map_count,
					temporary_working_buffer1,
					input_configuration_specific_list[0].feature_map_count,
					&beta,
					temporary_working_buffer2,
					output_configuration_specific.feature_map_count));

				cuda_util::transpose(
					*cuda_config,
					temporary_working_buffer2,
					*output_buffer,
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
					*output_buffer));
			}
		}

		void convolution_1x1_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				bias_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				1,
				output_configuration_specific.feature_map_count,
				1,
				1));
		}

		size_t convolution_1x1_layer_tester_cuda::get_temporary_working_per_entry_buffer_size() const
		{
			return (input_elem_count_per_entry_list[0] + output_elem_count_per_entry) * sizeof(float);
		}
	}
}
