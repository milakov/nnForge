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

#include "neuron_value_set_data_bunch_writer.h"

namespace nnforge
{
	neuron_value_set_data_bunch_writer::neuron_value_set_data_bunch_writer()
	{
	}

	neuron_value_set_data_bunch_writer::~neuron_value_set_data_bunch_writer()
	{
	}

	void neuron_value_set_data_bunch_writer::set_config_map(const std::map<std::string, layer_configuration_specific> config_map)
	{
		layer_name_to_config_and_value_set_map.clear();

		for(std::map<std::string, layer_configuration_specific>::const_iterator it = config_map.begin(); it != config_map.end(); ++it)
		{
			layer_name_to_config_and_value_set_map[it->first] = std::make_pair(
				it->second,
				neuron_value_set::ptr(new neuron_value_set(it->second.get_neuron_count())));
		}
	}

	void neuron_value_set_data_bunch_writer::write(
		unsigned int entry_id,
		const std::map<std::string, const float *>& data_map)
	{
		for(std::map<std::string, const float *>::const_iterator it = data_map.begin(); it != data_map.end(); ++it)
			layer_name_to_config_and_value_set_map[it->first].second->set_entry(entry_id, it->second);
	}
}
