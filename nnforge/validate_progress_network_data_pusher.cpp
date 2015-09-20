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

#include "validate_progress_network_data_pusher.h"

#include "neuron_value_set_data_bunch_writer.h"

#include <stdio.h>
#include <boost/format.hpp>

namespace nnforge
{
	validate_progress_network_data_pusher::validate_progress_network_data_pusher(
		forward_propagation::ptr forward_prop,
		structured_data_bunch_reader::ptr reader,
		unsigned int report_frequency)
		: forward_prop(forward_prop)
		, reader(reader)
		, report_frequency(report_frequency)
	{
	}

	validate_progress_network_data_pusher::~validate_progress_network_data_pusher()
	{
	}

	void validate_progress_network_data_pusher::push(
		const training_task_state& task_state,
		const network_schema& schema)
	{
		if ((task_state.get_current_epoch() % report_frequency) == 0)
		{
			forward_prop->set_data(*task_state.data);

			neuron_value_set_data_bunch_writer writer;
			forward_propagation::stat st = forward_prop->run(*reader, writer);

			forward_prop->clear_data();

			unsigned int last_index = static_cast<unsigned int>(task_state.history.size()) - 1;

			std::cout << "----- Validating -----" << std::endl;
			std::cout << st << std::endl;

			for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = writer.layer_name_to_config_and_value_set_map.begin(); it != writer.layer_name_to_config_and_value_set_map.end(); ++it)
				std::cout << schema.get_layer(it->first)->get_string_for_average_data(it->second.first, *it->second.second->get_average()) << std::endl;
		}
	}
}
