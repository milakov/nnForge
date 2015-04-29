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

#include "network_schema.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace nnforge
{
	// {51D12B28-F064-4F14-9AF1-F7FEA14AA141}
	const boost::uuids::uuid network_schema::schema_guid =
	{ 0x51, 0xd1, 0x2b, 0x28
	, 0xf0, 0x64
	, 0x4f, 0x14
	, 0x9a, 0xf1
	, 0xf7, 0xfe, 0xa1, 0x4a, 0xa1, 0x41 };

	network_schema::network_schema()
	{
	}

	const const_layer_list& network_schema::get_layers() const
	{
		return layers;
	}

	network_schema::operator const const_layer_list&() const
	{
		return get_layers();
	}

	void network_schema::add_layer(const_layer_smart_ptr new_layer)
	{
		layer_configuration new_config = new_layer->get_layer_configuration(output_config);

		layers.push_back(new_layer);

		output_config = new_config;
	}

	const boost::uuids::uuid& network_schema::get_uuid() const
	{
		return schema_guid;
	}

	void network_schema::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		const boost::uuids::uuid& network_guid = get_uuid();
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(network_guid.data), sizeof(network_guid.data));

		const const_layer_list& layers_to_write = get_layers();
		unsigned int layer_count = static_cast<unsigned int>(layers_to_write.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

		for(const_layer_list::const_iterator it = layers_to_write.begin(); it != layers_to_write.end(); ++it)
		{
			const boost::uuids::uuid& layer_guid = (*it)->get_uuid();
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(layer_guid.data), sizeof(layer_guid.data));

			(*it)->write(binary_stream_to_write_to);
		}

		binary_stream_to_write_to.flush();
	}

	std::vector<layer_data_configuration_list> network_schema::get_layer_data_configuration_list_list() const
	{
		std::vector<layer_data_configuration_list> res;

		for(const_layer_list::const_iterator it = layers.begin(); it != layers.end(); ++it)
			res.push_back((*it)->get_layer_data_configuration_list());

		return res;
	}

	void network_schema::read(std::istream& binary_stream_to_read_from)
	{
		clear();

		binary_stream_to_read_from.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid network_guid_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(network_guid_read.data), sizeof(network_guid_read.data));
		if (network_guid_read != get_uuid())
			throw neural_network_exception((boost::format("Unknown schema GUID encountered in input stream: %1%") % network_guid_read).str());

		unsigned int layer_count_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&layer_count_read), sizeof(layer_count_read));

		for(unsigned int i = 0; i < layer_count_read; ++i)
		{
			boost::uuids::uuid layer_read_guid;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(layer_read_guid.data), sizeof(layer_read_guid.data));

			layer_smart_ptr new_layer = single_layer_factory::get_const_instance().create_layer(layer_read_guid);
			new_layer->read(binary_stream_to_read_from, layer_read_guid);
			add_layer(new_layer);
		}
	}

	void network_schema::write_proto(std::ostream& stream_to_write_to) const
	{
		protobuf::NetworkSchema schema;
		if (!name.empty())
			schema.set_name(name);

		for(int i = 0; i < layers.size(); ++i)
		{
			protobuf::Layer * layer_proto = schema.add_layer();
			const_layer_smart_ptr current_layer = layers[i];
			layer_proto->set_type(current_layer->get_type_name());
			if (!current_layer->instance_name.empty())
				layer_proto->set_name(current_layer->instance_name);

			current_layer->write_proto(layer_proto);
		}

		google::protobuf::io::OstreamOutputStream output_stream(&stream_to_write_to);
		google::protobuf::TextFormat::Print(schema, &output_stream);
	}

	void network_schema::read_proto(std::istream& stream_to_read_from)
	{
		clear();

		protobuf::NetworkSchema schema;
		google::protobuf::io::IstreamInputStream input_stream(&stream_to_read_from);
		google::protobuf::TextFormat::Parse(&input_stream, &schema);

		name = schema.name();

		for(int i = 0; i < schema.layer_size(); ++i)
		{
			layer_smart_ptr new_layer = single_layer_factory::get_const_instance().create_layer(schema.layer(i).type());
			new_layer->instance_name = schema.layer(i).name();
			new_layer->read_proto(&schema.layer(i));
			add_layer(new_layer);
		}
	}

	void network_schema::clear()
	{
		layers.clear();
		output_config = layer_configuration();
	}

	layer_configuration_specific_list network_schema::get_layer_configuration_specific_list(
		const layer_configuration_specific& input_layer_configuration_specific) const
	{
		layer_configuration_specific_list res;

		res.push_back(input_layer_configuration_specific);

		for(unsigned int i = 0; i < layers.size(); ++i)
			res.push_back(layers[i]->get_output_layer_configuration_specific(res[i]));

		return res;
	}

	layer_configuration_specific_list network_schema::get_layer_configuration_specific_list_reverse(const layer_configuration_specific& output_layer_configuration_specific) const
	{
		layer_configuration_specific_list res;

		res.resize(layers.size() + 1);
		res[layers.size()] = output_layer_configuration_specific;

		for(unsigned int i = static_cast<unsigned int>(layers.size()); i > 0; --i)
			res[i - 1] = (layers[i - 1]->get_input_layer_configuration_specific(res[i]));

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > network_schema::get_input_rectangle_borders(
		const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		unsigned int output_layer_id) const
	{
		std::vector<std::pair<unsigned int, unsigned int> > input_rectangle_borders = output_rectangle_borders;

		for(int i = static_cast<int>(output_layer_id); i >= 0; --i)
			input_rectangle_borders = layers[i]->get_input_rectangle_borders(input_rectangle_borders);

		return input_rectangle_borders;
	}

	std::vector<unsigned int> network_schema::get_cumulative_tiling_factor_list(tiling_factor input_tiling_factor) const
	{
		std::vector<unsigned int> res;
		res.push_back(input_tiling_factor);

		tiling_factor current_tiling_factor = input_tiling_factor;
		for(unsigned int i = 0; i < layers.size(); ++i)
		{
			tiling_factor new_tf = layers[i]->get_tiling_factor();
			current_tiling_factor = current_tiling_factor * new_tf;
			res.push_back(current_tiling_factor);
		}

		return res;
	}
}

