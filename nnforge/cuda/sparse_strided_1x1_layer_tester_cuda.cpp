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

#include "sparse_strided_1x1_layer_tester_cuda.h"

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
		sparse_strided_1x1_layer_tester_cuda::sparse_strided_1x1_layer_tester_cuda()
			: output_data_desc(0)
			, bias_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_strided_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_converted_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&bias_desc));
		}

		sparse_strided_1x1_layer_tester_cuda::~sparse_strided_1x1_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(input_strided_data_desc);
			cudnnDestroyTensorDescriptor(input_converted_data_desc);
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyTensorDescriptor(bias_desc);
		}

		void sparse_strided_1x1_layer_tester_cuda::enqueue_forward_propagation(
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
			// Convert input data to packed NHWC format
			{
				cudnn_safe_call(cudnnSetStream(cuda_config->get_cudnn_handle(), stream_id));
				cudnn_util::set_tensor_descriptor(
					input_strided_data_desc,
					input_strided_config,
					entry_count,
					input_strides);
				cudnn_util::set_tensor_descriptor(
					input_converted_data_desc,
					input_strided_config,
					entry_count,
					input_converted_strides);
				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnTransformTensor(
					cuda_config->get_cudnn_handle(),
					&alpha,
					input_strided_data_desc,
					*input_buffers[0],
					&beta,
					input_converted_data_desc,
					*temporary_working_per_entry_buffer));
			}

			{
				cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
				float alpha = 1.0F;
				float beta = 0.0F;
				cusparseMatDescr_t mat_descr;
				cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
				cusparse_safe_call(cusparseScsrmm(
					cuda_config->get_cusparse_handle(),
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					output_configuration_specific.feature_map_count,
					entry_count * output_elem_count_per_feature_map,
					input_strided_config.feature_map_count,
					feature_map_connection_count,
					&alpha,
					mat_descr,
					*data[0],
					*data_custom[1],
					*data_custom[0],
					*temporary_working_per_entry_buffer,
					input_strided_config.feature_map_count,
					&beta,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					output_configuration_specific.feature_map_count));
			}

			// Convert output from NHWC to NCHW
			{
				cuda_util::transpose(
					*cuda_config,
					((float *)*temporary_working_per_entry_buffer) + input_converted_elem_count_per_entry_aligned * entry_count,
					*output_buffer,
					output_configuration_specific.feature_map_count,
					output_elem_count_per_feature_map,
					entry_count,
					stream_id);
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

		void sparse_strided_1x1_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;
			bias = layer_derived->bias;

			cudnn_util::set_tensor_bias_descriptor(
				bias_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));

			const std::vector<unsigned int>& strides = layer_derived->strides;
			input_strided_config.feature_map_count = input_configuration_specific_list[0].feature_map_count;
			input_strided_config.dimension_sizes = output_configuration_specific.dimension_sizes;
			input_strides.resize(strides.size() + 1);
			unsigned int dim_size = 1;
			for(int i = 0; i < strides.size(); ++i)
			{
				*(input_strides.rbegin() + i) = strides[i] * dim_size;
				dim_size *= input_configuration_specific_list[0].dimension_sizes[i];
			}
			input_strides[0] = dim_size;

			input_converted_strides.resize(strides.size() + 2);
			input_converted_strides[strides.size()] = 1;
			dim_size = input_strided_config.feature_map_count;
			for(int i = 0; i < strides.size(); ++i)
			{
				input_converted_strides[i] = dim_size;
				dim_size *= input_configuration_specific_list[0].dimension_sizes[i];
			}
			input_converted_strides.back() = dim_size;

			input_converted_elem_count_per_entry_aligned = (input_strided_config.get_neuron_count() + 4 - 1) / 4 * 4;
			output_elem_count_per_entry_aligned = (output_configuration_specific.get_neuron_count() + 4 - 1) / 4 * 4;
		}

		size_t sparse_strided_1x1_layer_tester_cuda::get_temporary_working_per_entry_buffer_size() const
		{
			return (input_converted_elem_count_per_entry_aligned * sizeof(float)) + (output_elem_count_per_entry_aligned * sizeof(float));
		}
	}
}
