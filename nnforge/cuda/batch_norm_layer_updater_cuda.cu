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

#include "batch_norm_layer_updater_cuda.h"

#include "neural_network_cudnn_exception.h"
#include "util_cuda.h"
#include "cudnn_util.h"
#include "../batch_norm_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void batch_norm_update_mean_invsigma_gradient_upd_kernel(
			float * __restrict gradient_mean,
			const float * __restrict target_mean,
			const float * __restrict current_mean,
			float * __restrict gradient_invvar,
			const float * __restrict target_invvar,
			const float * __restrict current_invvar,
			float mult,
			int elem_count)
		{
			int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
			if (elem_id < elem_count)
			{
				gradient_mean[elem_id] += mult * (target_mean[elem_id] - current_mean[elem_id]);
				gradient_invvar[elem_id] += mult * (target_invvar[elem_id] - current_invvar[elem_id]);
			}
		}

		const float batch_norm_layer_updater_cuda::mean_and_variance_gradient_slope = 1.0F; // As if it were MSE/2

		batch_norm_layer_updater_cuda::batch_norm_layer_updater_cuda()
		{
			cudnn_safe_call(cudnnCreateTensorDescriptor(&weights_desc));
			cudnn_safe_call(cudnnCreateTensorDescriptor(&data_desc));
		}

		batch_norm_layer_updater_cuda::~batch_norm_layer_updater_cuda()
		{
			cudnnDestroyTensorDescriptor(weights_desc);
			cudnnDestroyTensorDescriptor(data_desc);
		}

		void batch_norm_layer_updater_cuda::enqueue_forward_propagation(
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

			{
				cudnn_util::set_tensor_descriptor(
					data_desc,
					output_configuration_specific,
					entry_count);

				float alpha = 1.0F;
				float beta = 0.0F;
				cudnn_safe_call(cudnnBatchNormalizationForwardTraining(
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
					1.0,
					0,
					0,
					epsilon,
					((float *)*temporary_fixed_buffer),
					((float *)*temporary_fixed_buffer) + output_configuration_specific.feature_map_count));
			}
		}

		void batch_norm_layer_updater_cuda::enqueue_backward_data_and_weights_propagation(
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device::ptr> input_errors_buffers,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& gradient,
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

			{
				cudnn_util::set_tensor_descriptor(
					data_desc,
					output_configuration_specific,
					entry_count);

				float alpha_data = 1.0F;
				float beta_data = add_update_to_destination ? 1.0F : 0.0F;
				float alpha_weights = 1.0F;
				float beta_weights = 1.0F;
				cudnn_safe_call(cudnnBatchNormalizationBackward(
					cuda_config->get_cudnn_handle(),
					CUDNN_BATCHNORM_SPATIAL,
					&alpha_data,
					&beta_data,
					&alpha_weights,
					&beta_weights,
					data_desc,
					*input_neurons_buffers[0],
					data_desc,
					*output_errors_buffer,
					data_desc,
					*input_errors_buffers[0],
					weights_desc,
					*data[0],
					*gradient[0],
					*gradient[1],
					epsilon,
					((const float *)*temporary_fixed_buffer),
					((const float *)*temporary_fixed_buffer) + output_configuration_specific.feature_map_count));
			}

			{
				float mult = mean_and_variance_gradient_slope * static_cast<float>(entry_count);
				int elem_count = output_configuration_specific.feature_map_count;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);
				batch_norm_update_mean_invsigma_gradient_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*gradient[2],
					((const float *)*temporary_fixed_buffer),
					*data[2],
					*gradient[3],
					((const float *)*temporary_fixed_buffer) + output_configuration_specific.feature_map_count,
					*data[3],
					mult,
					elem_count);
			}
		}

		bool batch_norm_layer_updater_cuda::is_backward_data_and_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			return true;
		}

		bool batch_norm_layer_updater_cuda::is_backward_data_and_weights_dependent_on_output_buffer() const
		{
			return false;
		}

		void batch_norm_layer_updater_cuda::updater_configured()
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

		size_t batch_norm_layer_updater_cuda::get_temporary_fixed_buffer_size() const
		{
			return output_configuration_specific.feature_map_count * 2 * sizeof(float);
		}

		int batch_norm_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			if (action.get_action_type() == layer_action::backward_data_and_weights)
				return 0;
			else
				return -1;
		}
	}
}
