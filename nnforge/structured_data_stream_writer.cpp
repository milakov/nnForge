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

#include "structured_data_stream_writer.h"

#include "neural_network_exception.h"
#include "structured_data_stream_schema.h"

#include <boost/format.hpp>

namespace nnforge
{
	structured_data_stream_writer::structured_data_stream_writer(
		nnforge_shared_ptr<std::ostream> output_stream,
		const layer_configuration_specific& config)
		: out_stream(output_stream), entry_count(0)
	{
		out_stream->exceptions(std::ostream::failbit | std::ostream::badbit);

		neuron_count = config.get_neuron_count();

		out_stream->write(reinterpret_cast<const char*>(structured_data_stream_schema::structured_data_stream_guid.data), sizeof(structured_data_stream_schema::structured_data_stream_guid.data));

		config.write(*out_stream);

		entry_count_pos = out_stream->tellp();
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
	}

	structured_data_stream_writer::~structured_data_stream_writer()
	{
		// write entry count
		out_stream->seekp(entry_count_pos);
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		out_stream->flush();
	}

	void structured_data_stream_writer::write(const float * neurons)
	{
		out_stream->write(reinterpret_cast<const char*>(neurons), sizeof(float) * neuron_count);
		entry_count++;
	}
}
