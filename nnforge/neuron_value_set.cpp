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

#include <algorithm>
#include <functional>
#include <boost/format.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace nnforge
{
	// {B56F3DC8-E9CD-44A0-96AA-59C49CD60C72}
	const boost::uuids::uuid neuron_value_set::neuron_value_set_guid =
		{ 0xb5, 0x6f, 0x3d, 0xc8
		, 0xe9, 0xcd
		, 0x44, 0xa0
		, 0x96, 0xaa
		, 0x59, 0xc4, 0x9c, 0xd6, 0xc, 0x72 };

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

	const boost::uuids::uuid& neuron_value_set::get_uuid() const
	{
		return neuron_value_set_guid;
	}

	void neuron_value_set::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		const boost::uuids::uuid& guid = get_uuid();
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(guid.data), sizeof(guid.data));

		unsigned int entry_count = static_cast<unsigned int>(neuron_value_list.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&neuron_count), sizeof(neuron_count));

		for(unsigned int i = 0; i < entry_count; ++i)
		{
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&neuron_value_list[i]->at(0)), sizeof(float) * neuron_count);
		}

		binary_stream_to_write_to.flush();
	}

	void neuron_value_set::read(std::istream& binary_stream_to_read_from)
	{
		binary_stream_to_read_from.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		neuron_value_list.clear();

		boost::uuids::uuid data_guid_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(data_guid_read.data), sizeof(data_guid_read.data));
		if (data_guid_read != get_uuid())
			throw neural_network_exception((boost::format("Unknown output_neuron_value_set GUID encountered in input stream: %1%") % data_guid_read).str());

		unsigned int entry_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		binary_stream_to_read_from.read(reinterpret_cast<char*>(&neuron_count), sizeof(neuron_count));

		neuron_value_list.resize(entry_count);
		for(unsigned int i = 0; i < entry_count; ++i)
			neuron_value_list[i] = nnforge_shared_ptr<std::vector<float> >(new std::vector<float>(neuron_count));

		for(unsigned int i = 0; i < entry_count; ++i)
		{
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&neuron_value_list[i]->at(0)), sizeof(float) * neuron_count);
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
}
