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

#include "../layer.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_memobject.h"
#include "cuda_linear_buffer_device.h"
#include "../nn_types.h"

#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class layer_hessian_cuda
		{
		public:
			struct buffer_set
			{
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer;
				cuda_linear_buffer_device_smart_ptr input_errors_buffer;
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers;
			};

			virtual ~layer_hessian_cuda();

			void configure(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config,
				bool backprop_required);

			buffer_set allocate_all_buffers(unsigned int max_entry_count) const;

			void update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const;

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

			// input_errors_buffer is null if is_in_place_backprop() is true
			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_squared,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

			virtual void enqueue_update_hessian(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& hessian_data,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

		protected:
			layer_hessian_cuda();

			// The method is called when configuration is finished
			virtual void hessian_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const;

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			virtual bool is_in_place_backprop() const = 0;

			const_layer_smart_ptr layer_schema;
			cuda_running_configuration_const_smart_ptr cuda_config;

			layer_configuration_specific input_configuration_specific;
			layer_configuration_specific output_configuration_specific;

			bool backprop_required;

			unsigned int input_elem_count_per_entry;
			unsigned int output_elem_count_per_entry;
			unsigned int input_elem_count_per_feature_map;
			unsigned int output_elem_count_per_feature_map;

		private:
			layer_hessian_cuda(const layer_hessian_cuda&);
			layer_hessian_cuda& operator =(const layer_hessian_cuda&);
		};

		typedef nnforge_shared_ptr<layer_hessian_cuda> layer_hessian_cuda_smart_ptr;
	}
}
