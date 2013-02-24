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

#include "unsupervised_data_stream_reader.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	unsupervised_data_stream_reader_base::unsupervised_data_stream_reader_base(
		std::tr1::shared_ptr<std::istream> input_stream,
		size_t input_elem_size,
		unsigned int type_code)
		: in_stream(input_stream), input_elem_size(input_elem_size), entry_read_count(0)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != unsupervised_data_stream_schema::unsupervised_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown unsupervised data GUID encountered in input stream: %1%") % guid_read).str());

		input_configuration.read(*in_stream);

		input_neuron_count = input_configuration.get_neuron_count();

		unsigned int type_code_read;
		in_stream->read(reinterpret_cast<char*>(&type_code_read), sizeof(type_code_read));
		if (type_code_read != type_code)
			throw neural_network_exception((boost::format("Unexpected type code encountered in input stream: %1%") % type_code_read).str());

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	unsupervised_data_stream_reader_base::~unsupervised_data_stream_reader_base()
	{
	}

	void unsupervised_data_stream_reader_base::reset()
	{
		in_stream->seekg(reset_pos);

		entry_read_count = 0;
	}

	bool unsupervised_data_stream_reader_base::entry_available()
	{
		return (entry_read_count < entry_count);
	}

	void unsupervised_data_stream_reader_base::notify_read()
	{
		entry_read_count++;
	}
}
