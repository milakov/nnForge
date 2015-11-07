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
#include "varying_data_stream_writer.h"

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

		reset_pos = in_stream->tellg();

		in_stream->seekg(-static_cast<int>(sizeof(unsigned long long)) * entry_offsets.size(), std::ios::end);
		in_stream->read(reinterpret_cast<char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());
	}

	varying_data_stream_reader::~varying_data_stream_reader()
	{
	}

	bool varying_data_stream_reader::raw_read(
		unsigned int entry_id,
		std::vector<unsigned char>& all_elems)
	{
		if (entry_id >= entry_offsets.size() - 1)
			return false;

		unsigned long long total_entry_size = entry_offsets[entry_id + 1] - entry_offsets[entry_id];
		all_elems.resize(total_entry_size);
		{
			boost::lock_guard<boost::mutex> lock(read_data_from_stream_mutex);
			in_stream->seekg(reset_pos + (std::istream::off_type)(entry_offsets[entry_id]), std::ios::beg);
			in_stream->read(reinterpret_cast<char*>(&(*all_elems.begin())), total_entry_size);
		}

		return true;
	}

	int varying_data_stream_reader::get_entry_count() const
	{
		return static_cast<int>(entry_offsets.size() - 1);
	}

	raw_data_writer::ptr varying_data_stream_reader::get_writer(nnforge_shared_ptr<std::ostream> out) const
	{
		return raw_data_writer::ptr(new varying_data_stream_writer(out));
	}
}
