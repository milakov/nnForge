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

#include "error_function_updater_cuda.h"

#include "../neural_network_exception.h"

namespace nnforge
{
	namespace cuda
	{
		boost::uuids::uuid error_function_updater_cuda::empty_guid = boost::uuids::uuid();

		error_function_updater_cuda::error_function_updater_cuda()
		{
		}

		error_function_updater_cuda::~error_function_updater_cuda()
		{
		}

		void error_function_updater_cuda::enqueue_update_error_and_gradient_fused_with_activation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device_smart_ptr gradient_buffer,
			cuda_linear_buffer_device_smart_ptr error_buffer,
			const_cuda_linear_buffer_device_smart_ptr actual_output_buffer,
			const_cuda_linear_buffer_device_smart_ptr predicted_output_buffer,
			unsigned int offset_entry_id,
			unsigned int neuron_count,
			unsigned int updater_entry_count) const
		{
			throw neural_network_exception("enqueue_update_error_and_gradient_fused_with_activation not implemented for this error function");
		}

		const boost::uuids::uuid& error_function_updater_cuda::get_fusable_activation_uuid() const
		{
			return empty_guid;
		}
	}
}
