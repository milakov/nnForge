/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "neuron_value_set_data_bunch_reader.h"

#include <cstring>

namespace nnforge
{
	neuron_value_set_data_bunch_reader::neuron_value_set_data_bunch_reader(
		const std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >& layer_name_to_config_and_value_set_map)
		: layer_name_to_config_and_value_set_map(layer_name_to_config_and_value_set_map)
	{
	}

	std::map<std::string, layer_configuration_specific> neuron_value_set_data_bunch_reader::get_config_map() const
	{
		std::map<std::string, layer_configuration_specific> res;
		for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = layer_name_to_config_and_value_set_map.begin(); it != layer_name_to_config_and_value_set_map.end(); ++it)
			res.insert(std::make_pair(it->first, it->second.first));
		return res;
	}

	bool neuron_value_set_data_bunch_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		for(std::map<std::string, float *>::const_iterator it = data_map.begin(); it != data_map.end(); ++it)
		{
			const std::pair<layer_configuration_specific, neuron_value_set::ptr>& nvs = layer_name_to_config_and_value_set_map.find(it->first)->second;
			if (entry_id >= nvs.second->neuron_value_list.size())
				return false;
			std::shared_ptr<std::vector<float> > src_data = nvs.second->neuron_value_list[entry_id];
			float * src_ptr = &src_data->at(0);
			memcpy(it->second, src_ptr, nvs.first.get_neuron_count() * sizeof(float));
		}
		return true;
	}

	void neuron_value_set_data_bunch_reader::set_epoch(unsigned int epoch_id)
	{
	}

	int neuron_value_set_data_bunch_reader::get_entry_count() const
	{
		return static_cast<int>(layer_name_to_config_and_value_set_map.begin()->second.second->neuron_value_list.size());
	}

	structured_data_bunch_reader::ptr neuron_value_set_data_bunch_reader::get_narrow_reader(const std::set<std::string>& layer_names) const
	{
		std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> > narrow_layer_name_to_config_and_value_set_map;
		for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = layer_name_to_config_and_value_set_map.begin(); it != layer_name_to_config_and_value_set_map.end(); ++it)
		{
			if (layer_names.find(it->first) != layer_names.end())
				narrow_layer_name_to_config_and_value_set_map.insert(*it);
		}
		return neuron_value_set_data_bunch_reader::ptr(new neuron_value_set_data_bunch_reader(narrow_layer_name_to_config_and_value_set_map));
	}
}
