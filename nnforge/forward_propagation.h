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

#include "nn_types.h"
#include "network_schema.h"
#include "network_data.h"
#include "layer_configuration_specific.h"
#include "structured_data_bunch_reader.h"
#include "structured_data_bunch_writer.h"
#include "neuron_value_set.h"
#include "debug_state.h"
#include "network_action_schema.h"

#include <vector>
#include <string>
#include <set>
#include <map>
#include <ostream>

namespace nnforge
{
	class forward_propagation
	{
	public:
		class stat
		{
		public:
			float flops_per_entry;
			unsigned int entry_processed_count;
			float total_seconds;
		};

	public:
		typedef nnforge_shared_ptr<forward_propagation> ptr;

		~forward_propagation();

		void set_data(const network_data& data);

		void clear_data();

		// You don't need to call this method before calling test with supervised_data_reader
		void set_input_configuration_specific(const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map);

		stat run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer);

		bool is_schema_with_weights() const;

	protected:
		forward_propagation(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug);

		// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
		virtual void actual_set_data(network_data::const_ptr data) = 0;

		virtual void actual_clear_data() = 0;

		// schema, network data and data are guaranteed to be compatible
		// The function should return the number of entries processed
		virtual unsigned int actual_run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer) = 0;

		// The method is called when client calls set_input_configuration_specific and the configuration is modified.
		// The layer_config_map is guaranteed to be compatible with schema
		virtual void layer_config_map_modified() = 0;

	protected:
		network_schema::const_ptr schema;
		network_action_schema::ptr action_schema;
		std::vector<std::string> output_layer_names;
		debug_state::ptr debug;
		std::map<std::string, layer_configuration_specific> layer_config_map;
		std::map<std::string, unsigned int> cumulative_tiling_factor_map;
		float flops;
		std::set<std::string> data_layer_names;

	private:
		void update_flops();

		forward_propagation();
		forward_propagation(const forward_propagation&);
		forward_propagation& operator =(const forward_propagation&);
	};

	std::ostream& operator<< (std::ostream& out, const forward_propagation::stat& val);
}
