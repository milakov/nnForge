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

#include "network_data.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>
#include <numeric> 

namespace nnforge
{
	// {A8B18171-A294-4D99-B3B2-3A181374F226}
	const boost::uuids::uuid network_data::data_guid =
		{ 0xa8, 0xb1, 0x81, 0x71
		, 0xa2, 0x94
		, 0x4d, 0x99
		, 0xb3, 0xb2
		, 0x3a, 0x18, 0x13, 0x74, 0xf2, 0x26 };

	// {6D6CFB72-3A5C-4C5E-9566-029D2E649045}
	const boost::uuids::uuid network_data::data_guid_v1 =
		{ 0x6d, 0x6c, 0xfb, 0x72
		, 0x3a, 0x5c
		, 0x4c, 0x5e
		, 0x95, 0x66
		, 0x02, 0x9d, 0x2e, 0x64, 0x90, 0x45 };

	network_data::network_data()
	{
	}

	network_data::network_data(const const_layer_list& layer_list, float val)
		: data_list(layer_list, val)
		, data_custom_list(layer_list)
	{
	}

	network_data::network_data(const const_layer_list& layer_list, const network_data& other)
	{
		layer_data_list::const_iterator data_it = other.data_list.begin();
		layer_data_custom_list::const_iterator data_custom_it = other.data_custom_list.begin();
		for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			bool empty_data = (*it)->is_empty_data() && (*it)->is_empty_data_custom();
			if (empty_data)
			{
				data_list.push_back(layer_data_smart_ptr(new layer_data()));
				data_custom_list.push_back(layer_data_custom_smart_ptr(new layer_data_custom()));
			}
			else
			{
				if (data_it == other.data_list.end())
					throw neural_network_exception("data has less non-empty layers than schema does");
				if (data_custom_it == other.data_custom_list.end())
					throw neural_network_exception("custom data has less non-empty layers than schema does");

				while ((*data_it)->empty() && (*data_custom_it)->empty())
				{
					if (data_it != other.data_list.end())
						++data_it;
					else
						throw neural_network_exception("data has less non-empty layers than schema does");
					if (data_custom_it != other.data_custom_list.end())
						++data_custom_it;
					else
						throw neural_network_exception("custom data has less non-empty layers than schema does");
				}

				data_list.push_back(*data_it);
				data_custom_list.push_back(*data_custom_it);

				++data_it;
				++data_custom_it;
			}
		}
	}

	void network_data::check_network_data_consistency(const const_layer_list& layer_list) const
	{
		data_list.check_consistency(layer_list);
		data_custom_list.check_consistency(layer_list);
	}

	const boost::uuids::uuid& network_data::get_uuid() const
	{
		return data_guid;
	}

	void network_data::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		const boost::uuids::uuid& guid = get_uuid();
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(guid.data), sizeof(guid.data));

		unsigned int data_count = (unsigned int)data_list.size();
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&data_count), sizeof(data_count));

		for(unsigned int i = 0; i < data_count; ++i)
		{
			data_list[i]->write(binary_stream_to_write_to);
			data_custom_list[i]->write(binary_stream_to_write_to);
		}

		binary_stream_to_write_to.flush();
	}

	void network_data::read(std::istream& binary_stream_to_read_from)
	{
		binary_stream_to_read_from.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid data_guid_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(data_guid_read.data), sizeof(data_guid_read.data));
		bool read_data_custom = true;
		if (data_guid_read == data_guid_v1)
			read_data_custom = false;
		else if (data_guid_read != get_uuid())
			throw neural_network_exception((boost::format("Unknown data GUID encountered in input stream: %1%") % data_guid_read).str());

		unsigned int data_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&data_count), sizeof(data_count));

		data_list.resize(data_count);
		data_custom_list.resize(data_count);

		for(unsigned int i = 0; i < data_count; ++i)
		{
			data_list[i] = layer_data_smart_ptr(new layer_data());
			data_list[i]->read(binary_stream_to_read_from);

			data_custom_list[i] = layer_data_custom_smart_ptr(new layer_data_custom());
			if (read_data_custom)
				data_custom_list[i]->read(binary_stream_to_read_from);
		}
	}

	void network_data::randomize(
		const const_layer_list& layer_list,
		random_generator& gen)
	{
		for(unsigned int i = 0; i < layer_list.size(); ++i)
		{
			layer_list[i]->randomize_data(
				*data_list[i],
				*data_custom_list[i],
				gen);
		}
	}
}
