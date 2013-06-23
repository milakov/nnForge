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

#include "supervised_data_stream_writer.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	supervised_data_stream_writer::supervised_data_stream_writer(
		std::tr1::shared_ptr<std::ostream> output_stream,
		const layer_configuration_specific& input_configuration,
		const layer_configuration_specific& output_configuration)
		: out_stream(output_stream), entry_count(0), type_code(neuron_data_type::type_unknown)
	{
		out_stream->exceptions(std::ostream::failbit | std::ostream::badbit);

		input_neuron_count = input_configuration.get_neuron_count();
		output_neuron_count = output_configuration.get_neuron_count();

		out_stream->write(reinterpret_cast<const char*>(supervised_data_stream_schema::supervised_data_stream_guid.data), sizeof(supervised_data_stream_schema::supervised_data_stream_guid.data));

		input_configuration.write(*out_stream);

		output_configuration.write(*out_stream);

		type_code_pos = out_stream->tellp();
		out_stream->write(reinterpret_cast<const char*>(&type_code), sizeof(type_code));

		entry_count_pos = out_stream->tellp();
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
	}

	supervised_data_stream_writer::~supervised_data_stream_writer()
	{
		std::ostream::pos_type current_pos = out_stream->tellp();

		// write type code
		out_stream->seekp(type_code_pos);
		if (type_code == neuron_data_type::type_unknown)
			type_code = neuron_data_type::type_byte;
		unsigned int t = static_cast<unsigned int>(type_code);
		out_stream->write(reinterpret_cast<const char*>(&t), sizeof(t));

		// write entry count
		out_stream->seekp(entry_count_pos);
		out_stream->write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

		out_stream->seekp(current_pos);

		out_stream->flush();
	}

	void supervised_data_stream_writer::write(
		neuron_data_type::input_type type_code,
		const void * input_neurons,
		const float * output_neurons)
	{
		if (this->type_code == neuron_data_type::type_unknown)
		{
			this->type_code = type_code;
			input_elem_size = neuron_data_type::get_input_size(this->type_code);
		}
		else if (this->type_code != type_code)
			throw neural_network_exception((boost::format("Cannot write elements with different input type: %1% %2%") % this->type_code % type_code).str());

		out_stream->write(reinterpret_cast<const char*>(input_neurons), input_elem_size * input_neuron_count);
		out_stream->write(reinterpret_cast<const char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		entry_count++;
	}

	void supervised_data_stream_writer::write(
		const unsigned char * input_neurons,
		const float * output_neurons)
	{
		if (type_code == neuron_data_type::type_unknown)
		{
			type_code = neuron_data_type::type_byte;
			input_elem_size = neuron_data_type::get_input_size(type_code);
		}
		else if (type_code != neuron_data_type::type_byte)
			throw neural_network_exception((boost::format("Cannot write elements with different input type: %1% %2%") % type_code % neuron_data_type::type_byte).str());

		out_stream->write(reinterpret_cast<const char*>(input_neurons), input_elem_size * input_neuron_count);
		out_stream->write(reinterpret_cast<const char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		entry_count++;
	}

	void supervised_data_stream_writer::write(
		const float * input_neurons,
		const float * output_neurons)
	{
		if (type_code == neuron_data_type::type_unknown)
		{
			type_code = neuron_data_type::type_float;
			input_elem_size = neuron_data_type::get_input_size(type_code);
		}
		else if (type_code != neuron_data_type::type_float)
			throw neural_network_exception((boost::format("Cannot write elements with different input type: %1% %2%") % type_code % neuron_data_type::type_float).str());

		out_stream->write(reinterpret_cast<const char*>(input_neurons), input_elem_size * input_neuron_count);
		out_stream->write(reinterpret_cast<const char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		entry_count++;
	}
}
