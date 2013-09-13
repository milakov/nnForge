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
	unsupervised_data_stream_reader::unsupervised_data_stream_reader(std::tr1::shared_ptr<std::istream> input_stream)
		: in_stream(input_stream)
		, entry_read_count(0)
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
		type_code = static_cast<neuron_data_type::input_type>(type_code_read);

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	unsupervised_data_stream_reader::~unsupervised_data_stream_reader()
	{
	}

	void unsupervised_data_stream_reader::reset()
	{
		in_stream->seekg(reset_pos);

		entry_read_count = 0;
	}

	bool unsupervised_data_stream_reader::read(void * input_neurons)
	{
		if (!entry_available())
			return false;

		if (input_neurons)
			in_stream->read(reinterpret_cast<char*>(input_neurons), get_input_neuron_elem_size() * input_neuron_count);
		else
			in_stream->seekg(get_input_neuron_elem_size() * input_neuron_count, std::ios_base::cur);

		entry_read_count++;

		return true;
	}

	bool unsupervised_data_stream_reader::entry_available()
	{
		return (entry_read_count < get_entry_count());
	}

}
