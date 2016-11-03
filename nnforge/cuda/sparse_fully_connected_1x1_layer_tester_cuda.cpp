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

#include "sparse_fully_connected_1x1_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "cudnn_util.h"
#include "neural_network_cusparse_exception.h"
#include "neural_network_cudnn_exception.h"
#include "../sparse_convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		sparse_fully_connected_1x1_layer_tester_cuda::sparse_fully_connected_1x1_layer_tester_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		sparse_fully_connected_1x1_layer_tester_cuda::~sparse_fully_connected_1x1_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void sparse_fully_connected_1x1_layer_tester_cuda::enqueue_forward_propagation(
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
				cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				cusparseMatDescr_t mat_descr;
				cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
				cusparse_safe_call(cusparseScsrmm(
					cuda_config->get_cusparse_handle(),
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					output_elem_count_per_entry,
					entry_count,
					input_elem_count_per_entry_list[0],
					feature_map_connection_count,
					&alpha,
					mat_descr,
					*data[0],
					*data_custom[1],
					*data_custom[0],
					*input_buffers[0],
					input_elem_count_per_entry_list[0],
					&beta,
					*output_buffer,
					output_elem_count_per_entry));
			}

			// Add bias
			if (bias)
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					output_data_desc,
					output_configuration_specific,
					entry_count);
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

		void sparse_fully_connected_1x1_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const sparse_convolution_layer> layer_derived = std::dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;
			bias = layer_derived->bias;

			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));
		}
	}
}
