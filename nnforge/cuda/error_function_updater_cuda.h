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

#pragma once

#include <boost/uuid/uuid.hpp>

#include "../nn_types.h"

#include "cuda_running_configuration.h"
#include "cuda_linear_buffer_device.h"

namespace nnforge
{
	namespace cuda
	{
		class error_function_updater_cuda
		{
		public:
			virtual ~error_function_updater_cuda();

			virtual const boost::uuids::uuid& get_uuid() const = 0;

			virtual void enqueue_update_error_and_gradient(
				cudaStream_t stream_id,
				cuda_linear_buffer_device_smart_ptr gradient_buffer,
				cuda_linear_buffer_device_smart_ptr error_buffer,
				const_cuda_linear_buffer_device_smart_ptr actual_output_buffer,
				const_cuda_linear_buffer_device_smart_ptr predicted_output_buffer,
				unsigned int offset_entry_id,
				unsigned int neuron_count,
				unsigned int updater_entry_count) const = 0;

			virtual void enqueue_update_error_and_gradient_fused_with_activation(
				cudaStream_t stream_id,
				cuda_linear_buffer_device_smart_ptr gradient_buffer,
				cuda_linear_buffer_device_smart_ptr error_buffer,
				const_cuda_linear_buffer_device_smart_ptr actual_output_buffer,
				const_cuda_linear_buffer_device_smart_ptr predicted_output_buffer,
				unsigned int offset_entry_id,
				unsigned int neuron_count,
				unsigned int updater_entry_count) const;

			virtual const boost::uuids::uuid& get_fusable_activation_uuid() const;

		protected:
			error_function_updater_cuda();

		private:
			error_function_updater_cuda(const error_function_updater_cuda&);
			error_function_updater_cuda& operator =(const error_function_updater_cuda&);

			static boost::uuids::uuid empty_guid;
		};

		typedef nnforge_shared_ptr<error_function_updater_cuda> error_function_updater_cuda_smart_ptr;
		typedef nnforge_shared_ptr<const error_function_updater_cuda> const_error_function_updater_cuda_smart_ptr;
	}
}
