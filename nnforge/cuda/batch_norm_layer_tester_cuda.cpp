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

#include "batch_norm_layer_tester_cuda.h"

#include "neural_network_cudnn_exception.h"
#include "cudnn_util.h"
#include "../batch_norm_layer.h"

namespace nnforge
{
	namespace cuda
	{
		batch_norm_layer_tester_cuda::batch_norm_layer_tester_cuda()
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&weights_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&data_desc));
		}

		batch_norm_layer_tester_cuda::~batch_norm_layer_tester_cuda()
		{
			cudnnDestroyTensorDescriptor(weights_desc);
			cudnnDestroyTensorDescriptor(data_desc);
		}

		void batch_norm_layer_tester_cuda::enqueue_forward_propagation(
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

			{
				cudnn_util::set_tensor_descriptor(
					data_desc,
					output_configuration_specific,
					entry_count);

				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnBatchNormalizationForwardInference(
					cuda_config->get_cudnn_handle(),
					CUDNN_BATCHNORM_SPATIAL,
					&alpha,
					&beta,
					data_desc,
					*input_buffers[0],
					data_desc,
					*output_buffer,
					weights_desc,
					*data[0],
					*data[1],
					*data[2],
					*data[3],
					epsilon));
			}
		}

		void batch_norm_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const batch_norm_layer> layer_derived = nnforge_dynamic_pointer_cast<const batch_norm_layer>(layer_schema);

			epsilon = layer_derived->epsilon;
			if (epsilon < CUDNN_BN_MIN_EPSILON)
				throw neural_network_exception((boost::format("Too small epsilon specified: %1%, cuDNN requires at least %2%") % epsilon % CUDNN_BN_MIN_EPSILON).str());

			cudnn_util::set_tensor_bn_weights_descriptor(
				weights_desc,
				output_configuration_specific.feature_map_count,
				static_cast<unsigned int>(output_configuration_specific.dimension_sizes.size()));
		}
	}
}
