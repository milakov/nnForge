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

#include "neuron_value_set.h"

#include "neural_network_exception.h"

#include <cstring>
#include <algorithm>
#include <functional>
#include <boost/format.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace nnforge
{
	neuron_value_set::neuron_value_set(unsigned int neuron_count)
		: neuron_count(neuron_count)
	{
	}

	neuron_value_set::neuron_value_set(
		unsigned int neuron_count,
		unsigned int entry_count)
		: neuron_count(neuron_count)
		, neuron_value_list(entry_count)
	{
		for(unsigned int i = 0; i < entry_count; ++i)
			neuron_value_list[i] = nnforge_shared_ptr<std::vector<float> >(new std::vector<float>(neuron_count));
	}

	neuron_value_set::neuron_value_set(
		const std::vector<neuron_value_set::const_ptr>& source_neuron_value_set_list,
		merge_type_enum merge_type)
		: neuron_count(static_cast<unsigned int>(source_neuron_value_set_list[0]->neuron_value_list[0]->size()))
		, neuron_value_list(source_neuron_value_set_list[0]->neuron_value_list.size())
	{
		for(unsigned int i = 0; i < static_cast<unsigned int>(neuron_value_list.size()); ++i)
			neuron_value_list[i] = nnforge_shared_ptr<std::vector<float> >(new std::vector<float>(neuron_count));

		if (merge_type == merge_average)
		{
			float mult = 1.0F / static_cast<float>(source_neuron_value_set_list.size());
			for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
			{
				std::vector<float>& neuron_value_list_for_single_entry = *neuron_value_list[entry_id];
				for(unsigned int neuron_id = 0; neuron_id < neuron_value_list_for_single_entry.size(); neuron_id++)
				{
					float sum = 0.0F;
					for(std::vector<neuron_value_set::const_ptr>::const_iterator it = source_neuron_value_set_list.begin();
						it != source_neuron_value_set_list.end();
						it++)
					{
						sum += (*it)->neuron_value_list[entry_id]->at(neuron_id);
					}

					neuron_value_list[entry_id]->at(neuron_id) = sum * mult;
				}
			}
		}
		else if (merge_type == merge_median)
		{
			std::vector<float> val_list;
			for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
			{
				std::vector<float>& neuron_value_list_for_single_entry = *neuron_value_list[entry_id];
				for(unsigned int neuron_id = 0; neuron_id < neuron_value_list_for_single_entry.size(); neuron_id++)
				{
					val_list.clear();
					for(std::vector<neuron_value_set::const_ptr>::const_iterator it = source_neuron_value_set_list.begin();
						it != source_neuron_value_set_list.end();
						it++)
					{
						val_list.push_back((*it)->neuron_value_list[entry_id]->at(neuron_id));
					}
					std::sort(val_list.begin(), val_list.end());
					unsigned int elem_count = static_cast<unsigned int>(val_list.size());
					float val;
					if (elem_count & 1)
						val = val_list[elem_count >> 1];
					else
						val = (val_list[elem_count >> 1] + val_list[(elem_count >> 1) - 1]) * 0.5F;

					neuron_value_list[entry_id]->at(neuron_id) = val;
				}
			}
		}
	}

	neuron_value_set::neuron_value_set(const std::vector<std::pair<neuron_value_set::const_ptr, float> >& source_neuron_value_set_list)
		: neuron_count(static_cast<unsigned int>(source_neuron_value_set_list[0].first->neuron_value_list[0]->size()))
		, neuron_value_list(source_neuron_value_set_list[0].first->neuron_value_list.size())
	{
		for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
		{
			std::vector<float>& neuron_value_list_for_single_entry = *neuron_value_list[entry_id];
			for(unsigned int neuron_id = 0; neuron_id < neuron_value_list_for_single_entry.size(); neuron_id++)
			{
				float sum = 0.0F;
				for(std::vector<std::pair<neuron_value_set::const_ptr, float> >::const_iterator it = source_neuron_value_set_list.begin();
					it != source_neuron_value_set_list.end();
					it++)
				{
					sum += it->first->neuron_value_list[entry_id]->at(neuron_id) * it->second;
				}

				neuron_value_list[entry_id]->at(neuron_id) = sum;
			}
		}
	}

	void neuron_value_set::add_entry(const float * new_data)
	{
		neuron_value_list.push_back(nnforge_shared_ptr<std::vector<float> >(new std::vector<float>(neuron_count)));
		memcpy(&neuron_value_list.back()->at(0), new_data, neuron_count * sizeof(float));
	}

	nnforge_shared_ptr<std::vector<float> > neuron_value_set::get_average() const
	{
		std::vector<double> acc(neuron_count, 0.0);
		for(std::vector<nnforge_shared_ptr<std::vector<float> > >::const_iterator it = neuron_value_list.begin(); it != neuron_value_list.end(); ++it)
		{
			nnforge_shared_ptr<std::vector<float> > current_item = *it;
			for(unsigned int i = 0; i < neuron_count; ++i)
				acc[i] += static_cast<double>(current_item->at(i));
		}

		double mult = 1.0 / static_cast<double>(neuron_value_list.size());
		nnforge_shared_ptr<std::vector<float> > res(new std::vector<float>(neuron_count));
		for(unsigned int i = 0; i < neuron_count; ++i)
			res->at(i) = static_cast<float>(acc[i] * mult);

		return res;
	}

	void neuron_value_set::add(
		const neuron_value_set& other,
		float alpha,
		float beta)
	{
		for(unsigned int entry_id = 0; entry_id < neuron_value_list.size(); entry_id++)
		{
			const std::vector<float>& src = *other.neuron_value_list[entry_id];
			std::vector<float>& dst = *neuron_value_list[entry_id];
			for(unsigned int neuron_id = 0; neuron_id < neuron_count; ++neuron_id)
				dst[neuron_id] = alpha * dst[neuron_id] + beta * src[neuron_id];
		}
	}

	void neuron_value_set::compact(unsigned int sample_count)
	{
		if (sample_count == 1)
			return;

		unsigned int new_entry_count = static_cast<unsigned int>(neuron_value_list.size() / sample_count);
		if ((new_entry_count * sample_count) != neuron_value_list.size())
			throw neural_network_exception((boost::format("neuron_value_set::compact cannot operate on %1% entries no evenly divisible by sample count %2%") % neuron_value_list.size() % sample_count).str());

		float mult = 1.0F / static_cast<float>(sample_count);
		for(unsigned int dst_entry_id = 0; dst_entry_id < new_entry_count; ++dst_entry_id)
		{
			std::vector<float>& dst = *neuron_value_list[dst_entry_id];

			for(unsigned int neuron_id = 0; neuron_id < neuron_count; ++neuron_id)
			{
				float sum = 0.0F;
				for(unsigned int sample_id = 0; sample_id < sample_count; ++sample_id)
					sum += neuron_value_list[dst_entry_id * sample_count + sample_id]->at(neuron_id);
				dst[neuron_id] = mult * sum;
			}
		}

		neuron_value_list.resize(new_entry_count);
	}
}
