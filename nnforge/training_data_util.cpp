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

#include "training_data_util.h"

namespace nnforge
{
	void training_data_util::copy(
		const std::set<std::string>& layers_to_copy,
		structured_data_bunch_writer& writer,
		structured_data_bunch_reader& reader,
		int max_copy_elem_count)
	{
		std::map<std::string, layer_configuration_specific> config = reader.get_config_map();
		writer.set_config_map(config);

		std::map<std::string, std::vector<float> > data_buffer_map;
		std::map<std::string, float *> data_ptr_map;
		std::map<std::string, const float *> data_const_ptr_map;
		for(std::map<std::string, layer_configuration_specific>::const_iterator it = config.begin(); it != config.end(); ++it)
		{
			if (layers_to_copy.find(it->first) != layers_to_copy.end())
			{
				float * ptr = &data_buffer_map.insert(std::make_pair(it->first, std::vector<float>(it->second.get_neuron_count()))).first->second[0];
				data_ptr_map.insert(std::make_pair(it->first, ptr));
				data_const_ptr_map.insert(std::make_pair(it->first, ptr));
			}
		}

		int entry_copied_count = 0;
		while (((max_copy_elem_count < 0) || (entry_copied_count < max_copy_elem_count)) && reader.read(entry_copied_count, data_ptr_map))
		{
			writer.write(entry_copied_count, data_const_ptr_map);
			++entry_copied_count;
		}
	}
}
