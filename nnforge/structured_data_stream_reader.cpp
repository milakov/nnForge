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

#include "structured_data_stream_reader.h"

#include "neural_network_exception.h"
#include "structured_data_stream_schema.h"
#include "structured_data_stream_writer.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	structured_data_stream_reader::structured_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream)
		: in_stream(input_stream)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != structured_data_stream_schema::structured_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown structured data GUID encountered in input stream: %1%") % guid_read).str());

		input_configuration.read(*in_stream);

		input_neuron_count = input_configuration.get_neuron_count();

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	structured_data_stream_reader::~structured_data_stream_reader()
	{
	}

	bool structured_data_stream_reader::read(
		unsigned int entry_id,
		float * data)
	{
		if (entry_id >= entry_count)
			return false;

		{
			boost::lock_guard<boost::mutex> lock(read_data_from_stream_mutex);
			in_stream->seekg(reset_pos + (std::istream::off_type)entry_id * (std::istream::off_type)(sizeof(float) * input_neuron_count), std::ios::beg);
			in_stream->read(reinterpret_cast<char*>(data), sizeof(float) * input_neuron_count);
		}

		return true;
	}

	layer_configuration_specific structured_data_stream_reader::get_configuration() const
	{
		return input_configuration;
	}

	int structured_data_stream_reader::get_entry_count() const
	{
		return entry_count;
	}

	raw_data_writer::ptr structured_data_stream_reader::get_writer(nnforge_shared_ptr<std::ostream> out) const
	{
		return raw_data_writer::ptr(new structured_data_stream_writer(out, get_configuration()));
	}
}
