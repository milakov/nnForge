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

#include "../network_analyzer.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_stream.h"
#include "layer_updater_schema.h"

#include <utility>
#include <string>
#include <vector>

namespace nnforge
{
	namespace cuda
	{
		class network_analyzer_cuda : public network_analyzer
		{
		public:
			network_analyzer_cuda(
				network_schema_smart_ptr schema,
				cuda_running_configuration_const_smart_ptr cuda_config);

			virtual ~network_analyzer_cuda();

		protected:
			// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
			virtual void actual_set_data(network_data_smart_ptr data);

			// The method is called when client calls set_input_data. Input data is guaranteed to be compatible with schema
			virtual void actual_set_input_data(
				const void * input,
				neuron_data_type::input_type type_code);

			// The method is called when client calls run_backprop.
			// Output configuration (along with output_offset_list) is gueranteed to be compatible with specific configuration.
			virtual std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> actual_run_backprop(
				const layer_configuration_specific_snapshot& output_data,
				const std::vector<unsigned int>& output_offset_list,
				unsigned int output_layer_id,
				const std::vector<std::pair<unsigned int, unsigned int> >& input_rectangle_borders);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			network_analyzer_cuda(const network_analyzer_cuda&);
			network_analyzer_cuda& operator =(const network_analyzer_cuda&);

			void setup_network_cuda();

			void update_buffers_configuration_testing(buffer_cuda_size_configuration& buffer_configuration) const;

			cuda_running_configuration_const_smart_ptr cuda_config;

			cuda_stream_smart_ptr command_stream;

			const_layer_updater_schema_list updater_schemas;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > updater_schema_data;

			std::vector<layer_updater_cuda_smart_ptr> updater_list;
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > net_data;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > schema_data;

			cuda_linear_buffer_device_smart_ptr input_buf;
			cuda_linear_buffer_device_smart_ptr input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> > updater_input_and_all_buffers_pack;
			std::vector<cuda_linear_buffer_device_smart_ptr> output_errors_buffers;
		};
	}
}
