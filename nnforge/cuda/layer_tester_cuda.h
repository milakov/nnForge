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

#include "../layer.h"

#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_linear_buffer_device.h"

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class layer_tester_cuda
		{
		public:
			typedef std::shared_ptr<layer_tester_cuda> ptr;

			virtual ~layer_tester_cuda() = default;

			void configure(
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				layer::const_ptr layer_schema,
				cuda_running_configuration::const_ptr cuda_config);

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
				unsigned int entry_count) = 0;

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_data(layer_data::const_ptr host_data) const;

			virtual std::vector<cuda_linear_buffer_device::const_ptr> set_get_data_custom(layer_data_custom::const_ptr host_data);

			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_persistent_working_data() const;

			virtual int get_input_index_layer_can_write() const;

			// The function should return the minimum size and the flag indicating whether the tester would be happy to have larger working buffer
			virtual std::pair<size_t, bool> get_temporary_working_fixed_buffer_size() const;

			virtual size_t get_temporary_working_per_entry_buffer_size() const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

		protected:
			layer_tester_cuda() = default;

			// The method is called when configuration is finished
			virtual void tester_configured();

			virtual void notify_data_custom(layer_data_custom::const_ptr host_data_custom);

			layer::const_ptr layer_schema;
			cuda_running_configuration::const_ptr cuda_config;

			std::vector<layer_configuration_specific> input_configuration_specific_list;
			std::vector<unsigned int> input_elem_count_per_entry_list;
			std::vector<unsigned int> input_elem_count_per_feature_map_list;

			layer_configuration_specific output_configuration_specific;
			unsigned int output_elem_count_per_entry;
			unsigned int output_elem_count_per_feature_map;

		private:
			layer_tester_cuda(const layer_tester_cuda&) = delete;
			layer_tester_cuda& operator =(const layer_tester_cuda&) = delete;
		};
	}
}
