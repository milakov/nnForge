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

#include "gradient_modifier_layer_updater_cuda.h"

#include "../gradient_modifier_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		gradient_modifier_layer_updater_cuda::gradient_modifier_layer_updater_cuda()
		{
		}

		gradient_modifier_layer_updater_cuda::~gradient_modifier_layer_updater_cuda()
		{
		}

		void gradient_modifier_layer_updater_cuda::enqueue_forward_propagation(
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
			if ((const float *)(*input_buffers[0]) != (const float *)(*output_buffer))
			{
				cuda_util::copy_buffer(
					*cuda_config,
					*input_buffers[0],
					*output_buffer,
					output_elem_count_per_entry * entry_count,
					stream_id);
			}
		}

		void gradient_modifier_layer_updater_cuda::enqueue_backward_data_propagation(
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
			unsigned int entry_count)
		{
			cuda_util::multiply_by_value(
				*cuda_config,
				*input_errors_buffer,
				*output_errors_buffer,
				scale,
				output_elem_count_per_entry * entry_count,
				add_update_to_destination,
				stream_id);
		}

		void gradient_modifier_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const gradient_modifier_layer> layer_derived = nnforge_dynamic_pointer_cast<const gradient_modifier_layer>(layer_schema);

			scale = layer_derived->scale;
		}

		int gradient_modifier_layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			return 0;
		}

		bool gradient_modifier_layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			return false;
		}

		bool gradient_modifier_layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			return false;
		}
	}
}
