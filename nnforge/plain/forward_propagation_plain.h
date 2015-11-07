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

#include "../forward_propagation.h"
#include "plain_running_configuration.h"
#include "layer_tester_plain.h"

#include <map>

namespace nnforge
{
	namespace plain
	{
		class forward_propagation_plain : public forward_propagation
		{
		public:
			forward_propagation_plain(
				const network_schema& schema,
				const std::vector<std::string>& output_layer_names,
				debug_state::ptr debug,
				plain_running_configuration::const_ptr plain_config);

			virtual ~forward_propagation_plain();

		protected:
			// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
			virtual void actual_set_data(network_data::const_ptr data);

			virtual void actual_clear_data();

			// schema, network data and data are guaranteed to be compatible
			virtual unsigned int actual_run(
				structured_data_bunch_reader& reader,
				structured_data_bunch_writer& writer);

			// The method is called when client calls set_input_configuration_specific and the configuration is modified.
			// The layer_config_map is guaranteed to be compatible with schema
			virtual void layer_config_map_modified();

		private:
			void setup_dedicated_buffer_sizes();

			void setup_layer_buffer_sizes();

			void setup_temporary_working_fixed_buffer_sizes();

			void update_max_entry_count();

		private:
			plain_running_configuration::const_ptr plain_config;

			std::vector<layer_name_with_action> actions_in_execution_order;

			std::map<std::string, layer_tester_plain::const_ptr> testers;
			network_data::const_ptr net_data;

			size_t temporary_working_fixed_size;

			std::vector<size_t> layer_buffer_set_per_entry_size_list;
			std::map<layer_name_with_action, unsigned int> temporary_working_per_entry_data_action_to_set_map;
			std::map<layer_name_with_action, unsigned int> layer_buffer_action_to_set_map;

			std::map<std::string, size_t> dedicated_per_entry_data_name_to_size_map;

			unsigned int max_entry_count;

		private:
			static const unsigned int max_max_entry_count;

		private:
			forward_propagation_plain(const forward_propagation_plain&);
			forward_propagation_plain& operator =(const forward_propagation_plain&);
		};
	}
}
