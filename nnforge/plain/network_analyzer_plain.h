#pragma once
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
#include "plain_running_configuration.h"
#include "buffer_plain_size_configuration.h"
#include "layer_updater_plain.h"

namespace nnforge
{
	namespace plain
	{
		class network_analyzer_plain : public network_analyzer
		{
		public:
			network_analyzer_plain(
				network_schema_smart_ptr schema,
				plain_running_configuration_const_smart_ptr plain_config);

			~network_analyzer_plain();

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
			network_analyzer_plain(const network_analyzer_plain&);
			network_analyzer_plain& operator =(const network_analyzer_plain&);

			plain_running_configuration_const_smart_ptr plain_config;

			const_layer_updater_plain_list updater_list;
			network_data_smart_ptr data;
			additional_buffer_smart_ptr input_converted_buf;
			additional_buffer_smart_ptr initial_error_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> > input_buffer_and_additional_updater_buffers_pack;
			std::vector<additional_buffer_smart_ptr> output_errors_buffers;
		};
	}
}
