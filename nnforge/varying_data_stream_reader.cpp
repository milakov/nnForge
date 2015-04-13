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

#include "varying_data_stream_reader.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

#include "varying_data_stream_schema.h"
#include "neural_network_exception.h"

namespace nnforge
{
	varying_data_stream_reader::varying_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream)
		: in_stream(input_stream)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != varying_data_stream_schema::varying_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown varying data GUID encountered in input stream: %1%") % guid_read).str());

		unsigned int entry_count;
		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));
		entry_offsets.resize(entry_count + 1);

		in_stream->read(reinterpret_cast<char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());

		reset_pos = in_stream->tellg();
	}

	varying_data_stream_reader::~varying_data_stream_reader()
	{
	}

	void varying_data_stream_reader::reset()
	{
		rewind(0);
	}

	bool varying_data_stream_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		if (entry_read_count >= entry_offsets.size() - 1)
			return false;

		unsigned long long total_entry_size = entry_offsets[entry_read_count + 1] - entry_offsets[entry_read_count];
		all_elems.resize(total_entry_size);
		in_stream->read(reinterpret_cast<char*>(&(*all_elems.begin())), total_entry_size);

		++entry_read_count;

		return true;
	}

	void varying_data_stream_reader::rewind(unsigned int entry_id)
	{
		entry_read_count = entry_id;
		in_stream->seekg(reset_pos + (std::istream::off_type)(entry_offsets[entry_read_count]), std::ios::beg);
	}

	unsigned int varying_data_stream_reader::get_entry_count() const
	{
		return static_cast<unsigned int>(entry_offsets.size() - 1);
	}
}
