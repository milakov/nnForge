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

#include "../network_tester.h"
#include "layer_testing_schema.h"
#include "layer_tester_cuda.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_stream.h"

#include <utility>
#include <string>
#include <vector>

namespace nnforge
{
	namespace cuda
	{
		class network_tester_cuda : public network_tester
		{
		public:
			network_tester_cuda(
				network_schema_smart_ptr schema,
				cuda_running_configuration_const_smart_ptr cuda_config);

			virtual ~network_tester_cuda();

		protected:
			// schema, data and reader are guaranteed to be compatible
			virtual output_neuron_value_set_smart_ptr actual_run(unsupervised_data_reader& reader);

			// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
			virtual void actual_set_data(network_data_smart_ptr data);

			virtual void actual_clear_data();

			// The method is called when client calls get_snapshot. The data is guaranteed to be compatible with schema
			virtual std::vector<layer_configuration_specific_snapshot_smart_ptr> actual_get_snapshot(
				const void * input,
				neuron_data_type::input_type type_code);

			// The method is called when client calls get_snapshot. The data is guaranteed to be compatible with schema
			virtual layer_configuration_specific_snapshot_smart_ptr actual_run(
				const void * input,
				neuron_data_type::input_type type_code);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			network_tester_cuda(const network_tester_cuda&);
			network_tester_cuda& operator =(const network_tester_cuda&);

			void setup_network_cuda();

			void update_buffers_configuration_testing(buffer_cuda_size_configuration& buffer_configuration) const;

			void update_data();

			void init_cached_buffers();

			cuda_running_configuration_const_smart_ptr cuda_config;

			const_layer_testing_schema_list testing_schemas;

			network_data_smart_ptr host_net_data;

			cuda_stream_smart_ptr command_stream;
			cuda_stream_smart_ptr data_stream;

			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > net_data;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > net_data_custom;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > schema_data;

			std::vector<layer_tester_cuda_smart_ptr> tester_list;

			bool cached_buffers_initialized;
			cuda_linear_buffer_device_smart_ptr cached_input_buf;
			cuda_linear_buffer_device_smart_ptr cached_input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > cached_input_and_additional_buffers_pack;
			cuda_linear_buffer_device_smart_ptr cached_output_buffer;
		};
	}
}
