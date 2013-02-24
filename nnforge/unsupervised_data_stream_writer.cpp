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

#include "unsupervised_data_stream_writer.h"

namespace nnforge
{
	unsupervised_data_stream_writer_base::unsupervised_data_stream_writer_base(
		std::tr1::shared_ptr<std::ostream> output_stream,
		const layer_configuration_specific& input_configuration,
		unsigned int type_code)
		: out_stream(output_stream), entry_count(0)
	{
		out_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		input_neuron_count = input_configuration.get_neuron_count();

		out_stream->write(reinterpret_cast<const char*>(unsupervised_data_stream_schema::unsupervised_data_stream_guid.data), sizeof(unsupervised_data_stream_schema::unsupervised_data_stream_guid.data));

		input_configuration.write(*out_stream);

		out_stream->write(reinterpret_cast<const char*>(&type_code), sizeof(type_code));

		entry_count_pos = out_stream->tellp();

		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
	}

	unsupervised_data_stream_writer_base::~unsupervised_data_stream_writer_base()
	{
		// write entry count
		std::ostream::pos_type current_pos = out_stream->tellp();
		out_stream->seekp(entry_count_pos);
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
		out_stream->seekp(current_pos);

		out_stream->flush();
	}

	void unsupervised_data_stream_writer_base::write_output()
	{
		entry_count++;
	}
}
