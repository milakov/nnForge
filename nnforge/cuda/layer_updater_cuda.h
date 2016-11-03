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

#pragma once

#include "../layer.h"
#include "../layer_action.h"

#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_memobject.h"
#include "cuda_linear_buffer_device.h"

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class layer_updater_cuda
		{
		public:
			typedef std::shared_ptr<layer_updater_cuda> ptr;

			virtual ~layer_updater_cuda() = default;

			void configure(
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				layer::const_ptr layer_schema,
				cuda_running_configuration::const_ptr cuda_config,
				const std::set<layer_action>& actions);

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
				unsigned int entry_count) = 0;

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

			virtual void enqueue_backward_weights_propagation(
				cudaStream_t stream_id,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::ptr>& gradient,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
				cuda_linear_buffer_device::const_ptr output_errors_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
				cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
				unsigned int entry_count);

			virtual void enqueue_backward_data_and_weights_propagation(
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
				unsigned int entry_count);

			virtual std::vector<cuda_linear_buffer_device::ptr> get_data(layer_data::const_ptr host_data) const;

			virtual std::vector<cuda_linear_buffer_device::const_ptr> set_get_data_custom(layer_data_custom::const_ptr host_data);

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_persistent_working_data() const;

			void get_data_from_device(const std::vector<cuda_linear_buffer_device::ptr>& device_data, layer_data::ptr host_data) const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			// The function should return the minimum size and the flag indicating whether the tester would be happy to have larger working buffer
			virtual std::pair<size_t, bool> get_temporary_working_fixed_buffer_size(const layer_action& action) const;

			virtual size_t get_temporary_working_per_entry_buffer_size(const layer_action& action) const;

			// Created when doing forward prop and used for backward prop
			virtual size_t get_temporary_fixed_buffer_size() const;

			// Created when doing forward prop and used for backward prop
			virtual size_t get_temporary_per_entry_buffer_size() const;

			// Default impl returns -1
			virtual int get_input_index_layer_can_write(const layer_action& action) const;

			// Default impl returns true
			virtual bool is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const;

			// Default impl returns true
			virtual bool is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const;

			// Default impl returns get_temporary_fixed_buffer_size() != 0
			virtual bool is_backward_data_dependent_on_temporary_fixed_buffer(unsigned int action_input_index) const;

			// Default impl returns get_temporary_per_entry_buffer_size() != 0
			virtual bool is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const;

			// Default impl returns true
			virtual bool is_backward_data_and_weights_dependent_on_input_buffer(unsigned int data_input_index) const;

			// Default impl returns true
			virtual bool is_backward_data_and_weights_dependent_on_output_buffer() const;

			// Default impl returns get_temporary_fixed_buffer_size() != 0
			virtual bool is_backward_data_and_weights_dependent_on_temporary_fixed_buffer() const;

			// Default impl returns get_temporary_per_entry_buffer_size() != 0
			virtual bool is_backward_data_and_weights_dependent_on_temporary_per_entry_buffer() const;

			// Default impl returns true
			virtual bool is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const;

			// Default impl returns get_temporary_fixed_buffer_size() != 0
			virtual bool is_backward_weights_dependent_on_temporary_fixed_buffer() const;

			// Default impl returns get_temporary_per_entry_buffer_size() != 0
			virtual bool is_backward_weights_dependent_on_temporary_per_entry_buffer() const;

		protected:
			layer_updater_cuda() = default;

			// The method is called when configuration is finished
			virtual void updater_configured();

			virtual void notify_data_custom(layer_data_custom::const_ptr host_data_custom);

			layer::const_ptr layer_schema;
			cuda_running_configuration::const_ptr cuda_config;

			std::vector<layer_configuration_specific> input_configuration_specific_list;
			std::vector<unsigned int> input_elem_count_per_entry_list;
			std::vector<unsigned int> input_elem_count_per_feature_map_list;

			layer_configuration_specific output_configuration_specific;
			unsigned int output_elem_count_per_entry;
			unsigned int output_elem_count_per_feature_map;

			std::set<layer_action> actions;

		private:
			layer_updater_cuda(const layer_updater_cuda&) = delete;
			layer_updater_cuda& operator =(const layer_updater_cuda&) = delete;
		};
	}
}
