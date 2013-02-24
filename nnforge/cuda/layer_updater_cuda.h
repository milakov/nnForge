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

#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class layer_updater_cuda
		{
		public:
			struct buffer_set
			{
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer;
				cuda_linear_buffer_device_smart_ptr input_errors_buffer;
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers;
			};

			virtual ~layer_updater_cuda();

			void configure(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config,
				bool backprop_required,
				bool different_input);

			buffer_set allocate_all_buffers(unsigned int max_entry_count) const;

			void update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const;

			void update_buffer_configuration(
				buffer_cuda_size_configuration& buffer_configuration,
				unsigned int updater_entry_count) const;

			virtual void enqueue_test(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

			// input_errors_buffer is null if is_in_place_backprop() is true
			virtual void enqueue_backprop(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

			virtual void enqueue_update_weights(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& training_speed,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			void enqueue_forward_dropout(
				cudaStream_t stream_id,
				const_cuda_linear_buffer_device_smart_ptr random_buffer,
				cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				float dropout_rate,
				unsigned int mask,
				unsigned int entry_count,
				unsigned int offset_in_random_list);

			void enqueue_backward_dropout(
				cudaStream_t stream_id,
				const_cuda_linear_buffer_device_smart_ptr random_buffer,
				cuda_linear_buffer_device_smart_ptr input_errors_buffer,
				float dropout_rate,
				unsigned int mask,
				unsigned int entry_count,
				unsigned int offset_in_random_list);

		protected:
			layer_updater_cuda();

			// The method is called when configuration is finished
			virtual void updater_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			virtual bool is_in_place_backprop() const = 0;

			const_layer_smart_ptr layer_schema;
			cuda_running_configuration_const_smart_ptr cuda_config;

			layer_configuration_specific input_configuration_specific;
			layer_configuration_specific output_configuration_specific;

			bool backprop_required;
			bool different_input;

			unsigned int input_elem_count_per_entry;
			unsigned int output_elem_count_per_entry;
			unsigned int input_elem_count_per_feature_map;
			unsigned int output_elem_count_per_feature_map;

		private:
			layer_updater_cuda(const layer_updater_cuda&);
			layer_updater_cuda& operator =(const layer_updater_cuda&);
		};

		typedef std::tr1::shared_ptr<layer_updater_cuda> layer_updater_cuda_smart_ptr;
	}
}
