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

#pragma once

#include "layer_updater_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class rectified_linear_layer_updater_cuda : public layer_updater_cuda
		{
		public:
			rectified_linear_layer_updater_cuda() = default;

			virtual ~rectified_linear_layer_updater_cuda() = default;

			virtual void enqueue_forward_propagation(
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
				unsigned int entry_count);

			virtual void enqueue_backward_data_propagation(
				cudaStream_t stream_id,
				unsigned int input_index,
				cuda_linear_buffer_device::ptr input_errors_buffer,
				cuda_linear_buffer_device::const_ptr output_errors_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
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
				unsigned int entry_count);

			virtual int get_input_index_layer_can_write(const layer_action& action) const;

			virtual bool is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const;

			virtual bool is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const;

			virtual bool is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const;

			virtual size_t get_temporary_per_entry_buffer_size() const;

		protected:
			virtual void updater_configured();

		private:
			float negative_slope;
		};
	}
}
