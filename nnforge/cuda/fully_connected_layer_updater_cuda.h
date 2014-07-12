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

#include "layer_updater_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class fully_connected_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			fully_connected_layer_updater_cuda();

			virtual ~fully_connected_layer_updater_cuda();

			virtual void enqueue_test(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count);

			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count);

			virtual void enqueue_update_weights(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& gradient,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count);

		protected:
			virtual bool is_in_place_backprop() const;

		private:
			static int get_block_size(int entry_count);
		};
	}
}
