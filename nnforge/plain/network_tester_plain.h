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

#include "../network_tester.h"
#include "plain_running_configuration.h"
#include "layer_tester_plain.h"
#include "buffer_plain_size_configuration.h"

namespace nnforge
{
	namespace plain
	{
		class network_tester_plain : public network_tester
		{
		public:
			network_tester_plain(
				network_schema_smart_ptr schema,
				plain_running_configuration_const_smart_ptr plain_config);

			virtual ~network_tester_plain();

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
			network_tester_plain(const network_tester_plain&);
			network_tester_plain& operator =(const network_tester_plain&);

			void update_buffers_configuration_testing(buffer_plain_size_configuration& buffer_configuration) const;

			plain_running_configuration_const_smart_ptr plain_config;

			const_layer_tester_plain_list tester_list;
			network_data_smart_ptr net_data;
		};
	}
}
