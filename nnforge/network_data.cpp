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

#include "network_data.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>
#include <numeric> 

namespace nnforge
{
	// {6D6CFB72-3A5C-4C5E-9566-029D2E649045}
	const boost::uuids::uuid network_data::data_guid =
	{ 0x6d, 0x6c, 0xfb, 0x72
	, 0x3a, 0x5c
	, 0x4c, 0x5e
	, 0x95, 0x66
	, 0x02, 0x9d, 0x2e, 0x64, 0x90, 0x45 };

	network_data::network_data()
	{
	}

	network_data::network_data(const const_layer_list& layer_list, float val)
	{
		resize(layer_list.size());
		for(unsigned int i = 0; i < size(); ++i)
		{
			at(i) = layer_list[i]->create_layer_data();
			at(i)->fill(val);
		}
	}

	void network_data::check_network_data_consistency(const const_layer_list& layer_list) const
	{
		if (size() != layer_list.size())
			throw neural_network_exception("data count is not equal layer count");

		for(unsigned int i = 0; i < size(); ++i)
		{
			layer_list[i]->check_layer_data_consistency(*at(i));
		}
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

		unsigned int data_count = (unsigned int)size();
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&data_count), sizeof(data_count));

		for(layer_data_list::const_iterator it = begin(); it != end(); ++it)
		{
			(*it)->write(binary_stream_to_write_to);
		}

		binary_stream_to_write_to.flush();
	}

	void network_data::read(std::istream& binary_stream_to_read_from)
	{
		binary_stream_to_read_from.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid data_guid_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(data_guid_read.data), sizeof(data_guid_read.data));
		if (data_guid_read != get_uuid())
			throw neural_network_exception((boost::format("Unknown data GUID encountered in input stream: %1%") % data_guid_read).str());

		unsigned int data_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&data_count), sizeof(data_count));

		resize(data_count);

		for(unsigned int i = 0; i < size(); ++i)
		{
			at(i) = layer_data_smart_ptr(new layer_data());
			at(i)->read(binary_stream_to_read_from);
		}
	}

	void network_data::randomize(
		const const_layer_list& layer_list,
		random_generator& gen)
	{
		for(unsigned int i = 0; i < size(); ++i)
		{
			layer_list[i]->randomize_data(
				*at(i),
				gen);
		}
	}

	void network_data::fill(float val)
	{
		for(layer_data_list::iterator it = begin(); it != end(); ++it)
			(*it)->fill(val);
	}

	void network_data::random_fill(
		float min,
		float max,
		random_generator& gen)
	{
		for(layer_data_list::iterator it = begin(); it != end(); ++it)
			(*it)->random_fill(min, max, gen);
	}

	std::string network_data::get_stat() const
	{
		std::string stat = "";

		unsigned int layer_id = 0;
		for(layer_data_list::const_iterator it = begin(); it != end(); ++it)
		{
			std::string layer_stat;

			for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
			{
				const std::vector<float>& data = *it2;

				double sum = std::accumulate(data.begin(), data.end(), 0.0);
				float avg = sum / data.size();

				if (!layer_stat.empty())
					layer_stat += ", ";
				layer_stat += (boost::format("%|1$.5e|") % avg).str();
			}

			if (!layer_stat.empty())
			{
				if (!stat.empty())
					stat += ", ";
				stat += (boost::format("Layer %1% - (%2%)") % layer_id % layer_stat).str();
			}

			layer_id++;
		}

		return stat;
	}

	void network_data::apply_dropout_layer_config(
		const std::map<unsigned int, dropout_layer_config>& layer_id_to_dropout_config_map,
		bool is_direct)
	{
		for(std::map<unsigned int, dropout_layer_config>::const_iterator it = layer_id_to_dropout_config_map.begin(); it != layer_id_to_dropout_config_map.end(); ++it)
		{
			unsigned int layer_id = it->first;
			at(layer_id)->apply_dropout_layer_config(it->second, is_direct);
		}
	}
}
