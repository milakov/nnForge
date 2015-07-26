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
	// {4FF032B3-EA2B-481D-A278-FDA9AFEBE76D}
	const boost::uuids::uuid network_data::data_guid =
		{ 0x4f, 0xf0, 0x32, 0xb3
		, 0xea, 0x2b
		, 0x48, 0x1d
		, 0xa2, 0x78
		, 0xfd, 0xa9, 0xaf, 0xeb, 0xe7, 0x6d };

	// {A8B18171-A294-4D99-B3B2-3A181374F226}
	const boost::uuids::uuid network_data::data_guid_v2 =
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

	network_data::network_data(
		const std::vector<layer::const_ptr>& layer_list,
		float val)
		: data_list(layer_list, val)
		, data_custom_list(layer_list)
	{
	}

	network_data::network_data(
		const std::vector<layer::const_ptr>& layer_list,
		const network_data& other)
		: data_list(layer_list, other.data_list)
		, data_custom_list(layer_list, other.data_custom_list)
	{
	}

	void network_data::check_network_data_consistency(const std::vector<layer::const_ptr>& layer_list) const
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

		data_list.write(binary_stream_to_write_to);
		data_custom_list.write(binary_stream_to_write_to);

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

		if ((data_guid_read == data_guid_v1) || (data_guid_read == data_guid_v2))
		{
			read_legacy(binary_stream_to_read_from, read_data_custom);
			return;
		}
		else if (data_guid_read != get_uuid())
			throw neural_network_exception((boost::format("Unknown data GUID encountered in input stream: %1%") % data_guid_read).str());

		data_list.read(binary_stream_to_read_from);
		data_custom_list.read(binary_stream_to_read_from);
	}

	void network_data::read_legacy(
		std::istream& binary_stream_to_read_from,
		bool read_data_custom)
	{
		unsigned int data_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&data_count), sizeof(data_count));

		unsigned int layer_with_weights_layer_count = 0;
		for(unsigned int i = 0; i < data_count; ++i)
		{
			layer_data::ptr data(new layer_data());
			data->read(binary_stream_to_read_from);
			layer_data_custom::ptr data_custom(new layer_data_custom());
			if (read_data_custom)
				data_custom->read(binary_stream_to_read_from);

			if (data->empty() && data_custom->empty())
				continue;

			std::string instance_name = (boost::format("parameters_%1%") % layer_with_weights_layer_count).str();

			data_list.add(instance_name, data);
			data_custom_list.add(instance_name, data_custom);

			++layer_with_weights_layer_count;
		}
	}

	void network_data::randomize(
		const std::vector<layer::const_ptr>& layer_list,
		random_generator& gen)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer_data::ptr data = data_list.find((*it)->instance_name);
			layer_data_custom::ptr data_custom = data_custom_list.find((*it)->instance_name);
			(*it)->randomize_data(
				*data,
				*data_custom,
				gen);
		}
	}
}
