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
#include "training_momentum.h"
#include "network_action_schema.h"

#include <vector>
#include <string>
#include <set>
#include <map>
#include <ostream>

namespace nnforge
{
	class backward_propagation
	{
	public:
		class stat
		{
		public:
			float flops_per_entry;
			unsigned int entry_processed_count;
			float total_seconds;
			std::map<std::string, std::vector<float> > average_absolute_updates;
		};

	public:
		typedef nnforge_shared_ptr<backward_propagation> ptr;

		~backward_propagation();

		// You don't need to call this method before calling test with supervised_data_reader
		void set_input_configuration_specific(const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map);

		stat run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer,
			network_data& data,
			network_data::ptr momentum_data,
			const std::map<std::string, std::vector<float> >& learning_rates,
			unsigned int batch_size,
			float weight_decay,
			training_momentum momentum);

	protected:
		backward_propagation(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names,
			debug_state::ptr debug);

		// schema, network data and data are guaranteed to be compatible
		// The function should return the number of entries processed and average absolute updates
		virtual std::pair<unsigned int, std::map<std::string, std::vector<float> > > actual_run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer,
			network_data& data,
			network_data::ptr momentum_data,
			const std::map<std::string, std::vector<float> >& learning_rates,
			unsigned int batch_size,
			float weight_decay,
			training_momentum momentum) = 0;

		// The method is called when client calls set_input_configuration_specific and the configuration is modified.
		// The layer_config_map is guaranteed to be compatible with schema
		virtual void layer_config_map_modified() = 0;

	protected:
		network_schema::const_ptr schema;
		network_action_schema::const_ptr action_schema;
		std::vector<std::vector<layer_name_with_action> > same_output_action_sets;
		std::set<layer_name_with_action> add_output_actions;
		std::vector<std::string> output_layer_names;
		std::vector<std::string> error_source_layer_names;
		std::vector<std::string> exclude_data_update_layer_names;
		debug_state::ptr debug;
		std::map<std::string, layer_configuration_specific> layer_config_map;
		float flops;
		std::set<std::string> data_layer_names;

	private:
		void update_flops();

		backward_propagation();
		backward_propagation(const backward_propagation&);
		backward_propagation& operator =(const backward_propagation&);
	};

	std::ostream& operator<< (std::ostream& out, const backward_propagation::stat& val);
}
