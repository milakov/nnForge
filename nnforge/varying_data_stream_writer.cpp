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
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	varying_data_stream_writer::varying_data_stream_writer(
		nnforge_shared_ptr<std::ostream> output_stream,
		unsigned int entry_count)
		: out_stream(output_stream), entry_written_count(0), entry_offsets(entry_count + 1, 0)
	{
		out_stream->exceptions(std::ostream::failbit | std::ostream::badbit);

		out_stream->write(reinterpret_cast<const char*>(varying_data_stream_schema::varying_data_stream_guid.data), sizeof(varying_data_stream_schema::varying_data_stream_guid.data));

		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		entry_offsets_pos = out_stream->tellp();
		out_stream->write(reinterpret_cast<const char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());

		start_pos = out_stream->tellp();
		entry_offsets[0] = out_stream->tellp() - start_pos;
	}

	varying_data_stream_writer::~varying_data_stream_writer()
	{
		std::ostream::pos_type current_pos = out_stream->tellp();

		// write entry offsets
		out_stream->seekp(entry_offsets_pos);
		out_stream->write(reinterpret_cast<const char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());

		out_stream->seekp(current_pos);

		out_stream->flush();
	}

 	void varying_data_stream_writer::raw_write(
		const void * all_entry_data,
		size_t data_length)
	{
		if (entry_written_count >= entry_offsets.size() - 1)
			throw neural_network_exception((boost::format("Writing extra entries to supervised_varying_data_stream_writer not allowed, it was created with entry count %1%") % (entry_offsets.size() - 1)).str());

		out_stream->write(reinterpret_cast<const char*>(all_entry_data), data_length);

		++entry_written_count;

		entry_offsets[entry_written_count] = out_stream->tellp() - start_pos;
	}
}
