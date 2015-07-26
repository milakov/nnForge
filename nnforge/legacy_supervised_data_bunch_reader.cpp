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

#include "legacy_supervised_data_bunch_reader.h"

#include "neural_network_exception.h"
#include "supervised_data_stream_schema.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	legacy_supervised_data_bunch_reader::legacy_supervised_data_bunch_reader(
		nnforge_shared_ptr<std::istream> input_stream,
		const char * input_data_layer_name,
		const char * output_data_layer_name)
		: in_stream(input_stream)
		, input_data_layer_name(input_data_layer_name)
		, output_data_layer_name(output_data_layer_name)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != supervised_data_stream_schema::supervised_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown supervised data GUID encountered in input stream: %1%") % guid_read).str());

		input_configuration.read(*in_stream);
		output_configuration.read(*in_stream);

		unsigned int type_code_read;
		in_stream->read(reinterpret_cast<char*>(&type_code_read), sizeof(type_code_read));
		type_code = static_cast<neuron_data_type::input_type>(type_code_read);

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	legacy_supervised_data_bunch_reader::~legacy_supervised_data_bunch_reader()
	{
	}

	std::map<std::string, layer_configuration_specific> legacy_supervised_data_bunch_reader::get_config_map() const
	{
		std::map<std::string, layer_configuration_specific> res;

		res[input_data_layer_name] = input_configuration;
		res[output_data_layer_name] = output_configuration;

		return res;
	}

	bool legacy_supervised_data_bunch_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		if (entry_id >= entry_count)
			return false;

		{
			boost::unique_lock<boost::mutex> lock(read_data_mutex);

			in_stream->seekg(reset_pos + (std::istream::off_type)entry_id * (std::istream::off_type)((neuron_data_type::get_input_size(type_code) * input_configuration.get_neuron_count()) + (sizeof(float) * output_configuration.get_neuron_count())), std::ios::beg);
		
			std::map<std::string, float *>::const_iterator input_data_it = data_map.find(input_data_layer_name);
			if (input_data_it != data_map.end())
			{
				if (type_code == neuron_data_type::type_float)
					in_stream->read(reinterpret_cast<char*>(input_data_it->second), neuron_data_type::get_input_size(type_code) * input_configuration.get_neuron_count());
				else if (type_code == neuron_data_type::type_byte)
				{
					std::vector<unsigned char> buf(input_configuration.get_neuron_count());
					in_stream->read(reinterpret_cast<char*>(&buf[0]), neuron_data_type::get_input_size(type_code) * input_configuration.get_neuron_count());
					for(unsigned int i = 0; i < buf.size(); ++i)
						input_data_it->second[i] = static_cast<float>(buf[i]) * (1.0F / 255.0F);
				}
			}
			else
			{
				in_stream->seekg(neuron_data_type::get_input_size(type_code) * input_configuration.get_neuron_count(), std::ios_base::cur);
			}

			std::map<std::string, float *>::const_iterator output_data_it = data_map.find(output_data_layer_name);
			if (output_data_it != data_map.end())
			{
				in_stream->read(reinterpret_cast<char*>(output_data_it->second), sizeof(float) * output_configuration.get_neuron_count());
			}
		}

		return true;
	}

	void legacy_supervised_data_bunch_reader::next_epoch() const
	{
	}

	int legacy_supervised_data_bunch_reader::get_approximate_entry_count() const
	{
		return entry_count;
	}
}
