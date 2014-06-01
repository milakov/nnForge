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

#include "supervised_data_stream_reader.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	supervised_data_stream_reader::supervised_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream)
		: in_stream(input_stream)
		, entry_read_count(0)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != supervised_data_stream_schema::supervised_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown supervised data GUID encountered in input stream: %1%") % guid_read).str());

		input_configuration.read(*in_stream);
		output_configuration.read(*in_stream);

		input_neuron_count = input_configuration.get_neuron_count();
		output_neuron_count = output_configuration.get_neuron_count();

		unsigned int type_code_read;
		in_stream->read(reinterpret_cast<char*>(&type_code_read), sizeof(type_code_read));
		type_code = static_cast<neuron_data_type::input_type>(type_code_read);

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	supervised_data_stream_reader::~supervised_data_stream_reader()
	{
	}

	void supervised_data_stream_reader::reset()
	{
		in_stream->seekg(reset_pos);

		entry_read_count = 0;
	}

	bool supervised_data_stream_reader::read(
		void * input_neurons,
		float * output_neurons)
	{
		if (!entry_available())
			return false;

		if (input_neurons)
			in_stream->read(reinterpret_cast<char*>(input_neurons), get_input_neuron_elem_size() * input_neuron_count);
		else
			in_stream->seekg(get_input_neuron_elem_size() * input_neuron_count, std::ios_base::cur);

		if (output_neurons)
			in_stream->read(reinterpret_cast<char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		else
			in_stream->seekg(sizeof(*output_neurons) * output_neuron_count, std::ios_base::cur);

		entry_read_count++;

		return true;
	}

	bool supervised_data_stream_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		if (!entry_available())
			return false;

		size_t bytes_to_read = get_input_neuron_elem_size() * input_neuron_count + sizeof(float) * output_neuron_count;
		all_elems.resize(bytes_to_read);
		in_stream->read(reinterpret_cast<char*>(&(*all_elems.begin())), bytes_to_read);

		return true;
	}

	bool supervised_data_stream_reader::entry_available()
	{
		return (entry_read_count < entry_count);
	}

	void supervised_data_stream_reader::rewind(unsigned int entry_id)
	{
		in_stream->seekg(reset_pos + (std::istream::off_type)entry_id * (std::istream::off_type)((get_input_neuron_elem_size() * input_neuron_count) + (sizeof(float) * output_neuron_count)), std::ios::beg);

		entry_read_count = entry_id;
	}
}
