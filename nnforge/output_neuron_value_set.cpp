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

#include "output_neuron_value_set.h"

#include <algorithm>

namespace nnforge
{
	output_neuron_value_set::output_neuron_value_set()
	{
	}

	output_neuron_value_set::output_neuron_value_set(
		unsigned int entry_count,
		unsigned int neuron_count)
		: neuron_value_list(entry_count, std::vector<float>(neuron_count))
	{
	}

	output_neuron_value_set::output_neuron_value_set(
		const std::vector<std::tr1::shared_ptr<output_neuron_value_set> >& source_output_neuron_value_set_list,
		merge_type_enum merge_type)
		: neuron_value_list(source_output_neuron_value_set_list[0]->neuron_value_list.size(), std::vector<float>(source_output_neuron_value_set_list[0]->neuron_value_list[0].size()))
	{
		if (merge_type == merge_average)
		{
			float mult = 1.0F / static_cast<float>(source_output_neuron_value_set_list.size());
			for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
			{
				std::vector<float>& neuron_value_list_for_single_entry = neuron_value_list[entry_id];
				for(unsigned int neuron_id = 0; neuron_id < neuron_value_list_for_single_entry.size(); neuron_id++)
				{
					float sum = 0.0F;
					for(std::vector<output_neuron_value_set_smart_ptr>::const_iterator it = source_output_neuron_value_set_list.begin();
						it != source_output_neuron_value_set_list.end();
						it++)
					{
						sum += (*it)->neuron_value_list[entry_id][neuron_id];
					}

					neuron_value_list[entry_id][neuron_id] = sum * mult;
				}
			}
		}
		else if (merge_type == merge_median)
		{
			std::vector<float> val_list;
			for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
			{
				std::vector<float>& neuron_value_list_for_single_entry = neuron_value_list[entry_id];
				for(unsigned int neuron_id = 0; neuron_id < neuron_value_list_for_single_entry.size(); neuron_id++)
				{
					val_list.clear();
					for(std::vector<output_neuron_value_set_smart_ptr>::const_iterator it = source_output_neuron_value_set_list.begin();
						it != source_output_neuron_value_set_list.end();
						it++)
					{
						val_list.push_back((*it)->neuron_value_list[entry_id][neuron_id]);
					}
					std::sort(val_list.begin(), val_list.end());
					unsigned int elem_count = static_cast<unsigned int>(val_list.size());
					float val;
					if (elem_count & 1)
						val = val_list[elem_count >> 1];
					else
						val = (val_list[elem_count >> 1] + val_list[(elem_count >> 1) - 1]) * 0.5F;

					neuron_value_list[entry_id][neuron_id] = val;
				}
			}
		}
	}

	void output_neuron_value_set::clamp(
		float min_val,
		float max_val)
	{
		for(std::vector<std::vector<float> >::iterator it = neuron_value_list.begin(); it != neuron_value_list.end(); ++it)
		{
			for(std::vector<float>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
			{
				*it2 = std::max<float>(std::min<float>(*it2, max_val), min_val);
			}
		}
	}
}
