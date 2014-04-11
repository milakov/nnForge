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

#include "output_neuron_class_set.h"

#include <algorithm>
#include <map>

namespace nnforge
{
	output_neuron_class_set::output_neuron_class_set()
	{
	}

	output_neuron_class_set::output_neuron_class_set(const output_neuron_value_set& neuron_value_set)
		: class_id_list(neuron_value_set.neuron_value_list.size())
	{
		std::vector<unsigned int>::iterator dest_it = class_id_list.begin();
		for(std::vector<std::vector<float> >::const_iterator it = neuron_value_set.neuron_value_list.begin(); it != neuron_value_set.neuron_value_list.end(); it++)
		{
			const std::vector<float>& neuron_values = *it;

			unsigned int max_neuron_id = static_cast<unsigned int>(std::max_element(neuron_values.begin(), neuron_values.end()) - neuron_values.begin());

			*dest_it = max_neuron_id;
			dest_it++;
		}
	}

	output_neuron_class_set::output_neuron_class_set(const std::vector<nnforge_shared_ptr<output_neuron_class_set> >& source_output_neuron_class_set_list)
		: class_id_list(source_output_neuron_class_set_list[0]->class_id_list.size())
	{
		for(unsigned int i = 0; i < class_id_list.size(); i++)
		{
			std::map<unsigned int, unsigned int> dest_map;

			for(std::vector<output_neuron_class_set_smart_ptr>::const_iterator it = source_output_neuron_class_set_list.begin();
				it != source_output_neuron_class_set_list.end();
				it++)
			{
				unsigned int predicted_class_id = (*it)->class_id_list[i];

				std::map<unsigned int, unsigned int>::iterator it3 = dest_map.find(predicted_class_id);
				if (it3 == dest_map.end())
					dest_map.insert(std::make_pair(predicted_class_id, 1));
				else
					it3->second++;
			}

			unsigned int predicted_class_id = 0;
			unsigned int predicted_count = 0;
			for(std::map<unsigned int, unsigned int>::iterator it2 = dest_map.begin(); it2 != dest_map.end(); it2++)
			{
				unsigned int current_predicted_class_id = it2->first;
				unsigned int current_predicted_count = it2->second;

				if ((current_predicted_count > predicted_count) || ((current_predicted_count == predicted_count) && (current_predicted_class_id < predicted_class_id)))
				{
					predicted_class_id = current_predicted_class_id;
					predicted_count = current_predicted_count;
				}
			}

			class_id_list[i] = predicted_class_id;
		}
	}
}
