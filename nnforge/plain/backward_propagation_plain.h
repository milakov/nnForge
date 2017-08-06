/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "../backward_propagation.h"

#include "plain_running_configuration.h"
#include "layer_updater_plain.h"

#include <map>

namespace nnforge
{
	namespace plain
	{
		class backward_propagation_plain : public backward_propagation
		{
		public:
			backward_propagation_plain(
				const network_schema& schema,
				const std::vector<std::string>& output_layer_names,
				const std::vector<std::string>& error_source_layer_names,
				const std::vector<std::string>& exclude_data_update_layer_names,
				debug_state::ptr debug,
				profile_state::ptr profile,
				plain_running_configuration::const_ptr plain_config);

			virtual ~backward_propagation_plain() = default;

		protected:
			// schema, network data and data are guaranteed to be compatible
			// The function should set average absolute updates, the number of entries processed, and optionally time it takes to run each action
			virtual void actual_run(
				structured_data_bunch_reader& reader,
				structured_data_bunch_writer& writer,
				network_data& data,
				network_data::ptr momentum_data,
				network_data::ptr momentum_data2,
				const std::map<std::string, std::vector<float> >& learning_rates,
				unsigned int batch_size,
				unsigned int max_chunk_size,
				float weight_decay,
				training_momentum momentum,
				unsigned int epoch_id,
				std::map<std::string, std::vector<float> >& average_absolute_updates,
				unsigned int& entries_processed,
				std::map<layer_name_with_action, float>& action_seconds,
				float& idle_seconds);

			// The method is called when client calls set_input_configuration_specific and the configuration is modified.
			// The layer_config_map is guaranteed to be compatible with schema
			virtual void layer_config_map_modified();

		private:
			void setup_dedicated_buffer_sizes();

			void setup_layer_buffer_sizes();

			void setup_temporary_working_fixed_buffer_sizes();

			void update_buffer_config();

			void apply_gradient(
				const std::string& layer_name,
				layer_data::ptr data,
				layer_data::ptr gradient,
				layer_data::ptr previous_upd,
				layer_data::ptr previous_upd2,
				std::vector<double>& updates_accumulated,
				const std::vector<float>& learning_rates,
				float normalizer,
				float weight_decay,
				training_momentum momentum,
				unsigned int iteration_id) const;

		private:
			plain_running_configuration::const_ptr plain_config;

			std::vector<layer_name_with_action> actions_in_execution_order;
			std::map<std::string, std::set<layer_action> > layer_name_to_action_set_map;

			std::map<std::string, layer_updater_plain::const_ptr> updaters;

			size_t temporary_working_fixed_size;

			std::vector<size_t> layer_buffer_set_per_entry_size_list;
			std::map<layer_name_with_action, unsigned int> temporary_working_per_entry_data_action_to_set_map;
			std::map<layer_name_with_action, unsigned int> layer_buffer_action_to_set_map;
			std::map<layer_name_with_action, unsigned int> temporary_per_entry_data_action_to_set_map;

			std::map<std::string, size_t> dedicated_per_entry_data_name_to_size_map;

			buffer_plain_size_configuration buffer_config_without_data_and_momentum;

		private:
			backward_propagation_plain(const backward_propagation_plain&) = delete;
			backward_propagation_plain& operator =(const backward_propagation_plain&) = delete;
		};
	}
}
