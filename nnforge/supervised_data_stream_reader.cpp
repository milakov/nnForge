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
	supervised_data_stream_reader_base::supervised_data_stream_reader_base(
		std::tr1::shared_ptr<std::istream> input_stream,
		size_t input_elem_size,
		unsigned int type_code)
		: in_stream(input_stream), input_elem_size(input_elem_size), entry_read_count(0)
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
		if (type_code_read != type_code)
			throw neural_network_exception((boost::format("Unexpected type code encountered in input stream: %1%") % type_code_read).str());

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	supervised_data_stream_reader_base::~supervised_data_stream_reader_base()
	{
	}

	void supervised_data_stream_reader_base::reset()
	{
		in_stream->seekg(reset_pos);

		entry_read_count = 0;
	}

	void supervised_data_stream_reader_base::rewind(unsigned int entry_id)
	{
		in_stream->seekg(reset_pos);
		in_stream->seekg((std::istream::off_type)entry_id * (std::istream::off_type)((input_elem_size * input_neuron_count) + (sizeof(float) * output_neuron_count)), std::ios::cur);

		entry_read_count = entry_id;
	}

	void supervised_data_stream_reader_base::read_output(float * output_neurons)
	{
		if (output_neurons)
			in_stream->read(reinterpret_cast<char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		else
			in_stream->seekg(sizeof(*output_neurons) * output_neuron_count, std::ios_base::cur);

		entry_read_count++;
	}

	bool supervised_data_stream_reader_base::entry_available()
	{
		return (entry_read_count < entry_count);
	}

	randomized_classifier_keeper::randomized_classifier_keeper()
		: pushed_count(0)
		, remaining_ratio(0.0F)
	{
	}

	bool randomized_classifier_keeper::is_empty()
	{
		return entry_id_list.empty();
	}

	float randomized_classifier_keeper::get_ratio()
	{
		return remaining_ratio;
	}

	void randomized_classifier_keeper::push(unsigned int entry_id)
	{
		entry_id_list.push_back(entry_id);
		++pushed_count;

		update_ratio();
	}

	unsigned int randomized_classifier_keeper::peek_random(random_generator& rnd)
	{
		std::tr1::uniform_int<unsigned int> dist(0, static_cast<unsigned int>(entry_id_list.size()) - 1);

		unsigned int index = dist(rnd);
		unsigned int entry_id = entry_id_list[index];

		unsigned int leftover_entry_id = entry_id_list[entry_id_list.size() - 1];
		entry_id_list[index] = leftover_entry_id;

		entry_id_list.pop_back();

		update_ratio();

		return entry_id;
	}

	void randomized_classifier_keeper::update_ratio()
	{
		remaining_ratio = pushed_count > 0 ? static_cast<float>(entry_id_list.size()) / static_cast<float>(pushed_count) : 0.0F;
	}
}
