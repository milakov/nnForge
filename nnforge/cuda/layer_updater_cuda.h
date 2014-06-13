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
#include "../nn_types.h"

#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_memobject.h"
#include "cuda_linear_buffer_device.h"

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
				// dynamic memobject list is intendent to store shallow, lighweight objects, for example, texture objects
				std::vector<cuda_memobject_smart_ptr> dynamic_memobjects;
			};

			virtual ~layer_updater_cuda();

			void configure(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config,
				bool backprop_required,
				bool different_input);

			buffer_set allocate_all_buffers(unsigned int max_entry_count);

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
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
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
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count) = 0;

			virtual void enqueue_update_weights(
				unsigned int offset_input_entry_id,
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& learning_rate,
				cuda_linear_buffer_device_smart_ptr output_errors_buffer,
				const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
				unsigned int entry_count,
				float weight_decay);

			virtual std::vector<unsigned int> get_incoming_weight_count_per_output_neuron_list() const;

			std::vector<cuda_linear_buffer_device_smart_ptr> get_data(const std::vector<layer_data_smart_ptr>& host_data_list) const;

			std::vector<const_cuda_linear_buffer_device_smart_ptr> get_learning_rate(const std::vector<const_layer_data_smart_ptr>& host_learning_rate_list) const;

			void get_data_from_device(const std::vector<cuda_linear_buffer_device_smart_ptr>& device_data, std::vector<layer_data_smart_ptr>& host_data) const;

		protected:
			layer_updater_cuda();

			// The method is called when configuration is finished
			virtual void updater_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const;

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const;

			virtual void set_max_entry_count(unsigned int max_entry_count);

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			virtual int get_dynamic_memobject_count() const;

			virtual bool is_in_place_backprop() const = 0;

			virtual unsigned int get_data_elem_count(unsigned int part_id, unsigned int source_elem_count) const;

			virtual void fill_data_for_device(
				unsigned int part_id,
				const float * src,
				float * dst,
				unsigned int count) const;

			virtual void fill_data_for_host(
				unsigned int part_id,
				const float * src,
				float * dst,
				unsigned int count) const;

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

		typedef nnforge_shared_ptr<layer_updater_cuda> layer_updater_cuda_smart_ptr;
	}
}
