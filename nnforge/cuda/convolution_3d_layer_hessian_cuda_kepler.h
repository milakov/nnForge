/*
 *  Copyright 2011-2013 Maxim Milakov
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

#pragma once

#include "layer_hessian_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class convolution_3d_layer_hessian_cuda_kepler : public layer_hessian_cuda
		{
		public:
			convolution_3d_layer_hessian_cuda_kepler();

			virtual ~convolution_3d_layer_hessian_cuda_kepler();

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_squared,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			virtual void enqueue_update_hessian(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& hessian_data,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

		protected:
			virtual bool is_in_place_backprop() const;

			virtual void hessian_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			std::vector<int> window_sizes;

		private:
			static int get_block_size(int width);

			static int get_bias_update_block_size(int entry_count);

			static int get_weights_update_block_size(int entry_count);
		};
	}
}
