/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "varying_data_stream_writer.h"

#include "varying_data_stream_schema.h"

#include <boost/format.hpp>

namespace nnforge
{
	varying_data_stream_writer::varying_data_stream_writer(nnforge_shared_ptr<std::ostream> output_stream)
		: out_stream(output_stream), entry_offsets(1, 0)
	{
		out_stream->exceptions(std::ostream::failbit | std::ostream::badbit);

		out_stream->write(reinterpret_cast<const char*>(varying_data_stream_schema::varying_data_stream_guid.data), sizeof(varying_data_stream_schema::varying_data_stream_guid.data));

		entry_count_pos = out_stream->tellp();
		unsigned int entry_count = 0;
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		reset_pos = out_stream->tellp();
	}

	varying_data_stream_writer::~varying_data_stream_writer()
	{
		// write entry offsets
		out_stream->write(reinterpret_cast<const char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());

		// write entry count
		out_stream->seekp(entry_count_pos);
		unsigned int entry_count = static_cast<unsigned int>(entry_offsets.size() - 1);
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		out_stream->flush();
	}

 	void varying_data_stream_writer::raw_write(
		const void * all_entry_data,
		size_t data_length)
	{
		out_stream->write(reinterpret_cast<const char*>(all_entry_data), data_length);
		entry_offsets.push_back(entry_offsets.back() + data_length);
	}
}
