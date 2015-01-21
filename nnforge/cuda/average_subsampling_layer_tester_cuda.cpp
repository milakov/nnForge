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

#include "average_subsampling_layer_tester_cuda.h"

#include "neural_network_cudnn_exception.h"

#include "../average_subsampling_layer.h"

namespace nnforge
{
	namespace cuda
	{
		average_subsampling_layer_tester_cuda::average_subsampling_layer_tester_cuda()
			: input_data_desc(0)
			, output_data_desc(0)
			, subsampling_desc(0)
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&input_data_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&output_data_desc));
			cudnn_safe_call(cudnnCreatePoolingDescriptor(&subsampling_desc));
		}

		average_subsampling_layer_tester_cuda::~average_subsampling_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(input_data_desc);
			cudnnDestroyTensorDescriptor(output_data_desc);
			cudnnDestroyPoolingDescriptor(subsampling_desc);
		}

		void average_subsampling_layer_tester_cuda::enqueue_test(
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
			cudnn_safe_call(cudnnSetTensor4dDescriptor(
				output_data_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				entry_count,
				output_configuration_specific.feature_map_count,
				(output_configuration_specific.dimension_sizes.size() > 1) ? output_configuration_specific.dimension_sizes[1] : 1,
				output_configuration_specific.dimension_sizes[0]));

			float alpha = 1.0F;
			float beta = 0.0F;
			cudnn_safe_call(cudnnPoolingForward(
				cuda_config->get_cudnn_handle(),
				subsampling_desc,
				&alpha,
				input_data_desc,
				*input_buffer,
				&beta,
				output_data_desc,
				*additional_buffers[0]));
		}

		cuda_linear_buffer_device_smart_ptr average_subsampling_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		void average_subsampling_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const average_subsampling_layer> layer_derived = nnforge_dynamic_pointer_cast<const average_subsampling_layer>(layer_schema);

			cudnn_safe_call(cudnnSetPooling2dDescriptor(
				subsampling_desc,
				CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
				(layer_derived->subsampling_sizes.size() > 1) ? layer_derived->subsampling_sizes[1] : 1,
				layer_derived->subsampling_sizes[0],
				0,
				0,
				(layer_derived->subsampling_sizes.size() > 1) ? layer_derived->subsampling_sizes[1] : 1,
				layer_derived->subsampling_sizes[0]));
		}

		std::vector<size_t> average_subsampling_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}
	}
}
