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

#include "rgb_to_yuv_convert_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	// {FEEAE5CB-935C-4CF9-AEFD-9D0F2F5F60F3}
	const boost::uuids::uuid rgb_to_yuv_convert_layer::layer_guid =
	{ 0xfe, 0xea, 0xe5, 0xcb
	, 0x93, 0x5c
	, 0x4c, 0xf9
	, 0xae, 0xfd
	, 0x9d, 0xf, 0x2f, 0x5f, 0x60, 0xf3 };

	color_feature_map_config::color_feature_map_config()
	{
	}

	color_feature_map_config::color_feature_map_config(
		unsigned int red_and_y_feature_map_id,
		unsigned int green_and_u_feature_map_id,
		unsigned int blue_and_v_feature_map_id)
		: red_and_y_feature_map_id(red_and_y_feature_map_id)
		, green_and_u_feature_map_id(green_and_u_feature_map_id)
		, blue_and_v_feature_map_id(blue_and_v_feature_map_id)
	{
	}

	rgb_to_yuv_convert_layer::rgb_to_yuv_convert_layer(const std::vector<color_feature_map_config>& color_feature_map_config_list)
		: color_feature_map_config_list(color_feature_map_config_list)
	{
		if (color_feature_map_config_list.empty())
			throw neural_network_exception("Configuration list for RGB to YUV conversion layer may not be empty");
	}

	const boost::uuids::uuid& rgb_to_yuv_convert_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr rgb_to_yuv_convert_layer::clone() const
	{
		return layer_smart_ptr(new rgb_to_yuv_convert_layer(*this));
	}

	layer_configuration_specific rgb_to_yuv_convert_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		for(std::vector<color_feature_map_config>::const_iterator it = color_feature_map_config_list.begin(); it != color_feature_map_config_list.end(); ++it)
		{
			if (it->red_and_y_feature_map_id >= input_configuration_specific.feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for red and Y (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->red_and_y_feature_map_id % input_configuration_specific.feature_map_count).str());
			if (it->green_and_u_feature_map_id >= input_configuration_specific.feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for green and U (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->green_and_u_feature_map_id % input_configuration_specific.feature_map_count).str());
			if (it->blue_and_v_feature_map_id >= input_configuration_specific.feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for blue and V (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->blue_and_v_feature_map_id % input_configuration_specific.feature_map_count).str());
		}

		return layer_configuration_specific(input_configuration_specific);
	}

	void rgb_to_yuv_convert_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int color_feature_map_config_count = static_cast<unsigned int>(color_feature_map_config_list.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&color_feature_map_config_count), sizeof(color_feature_map_config_count));

		for(std::vector<color_feature_map_config>::const_iterator it = color_feature_map_config_list.begin(); it != color_feature_map_config_list.end(); it++)
		{
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&it->red_and_y_feature_map_id), sizeof(it->red_and_y_feature_map_id));
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&it->green_and_u_feature_map_id), sizeof(it->green_and_u_feature_map_id));
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&it->blue_and_v_feature_map_id), sizeof(it->blue_and_v_feature_map_id));
		}
	}

	void rgb_to_yuv_convert_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		unsigned int color_feature_map_config_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&color_feature_map_config_count), sizeof(color_feature_map_config_count));
		color_feature_map_config_list.resize(color_feature_map_config_count);

		for(std::vector<color_feature_map_config>::iterator it = color_feature_map_config_list.begin(); it != color_feature_map_config_list.end(); it++)
		{
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&it->red_and_y_feature_map_id), sizeof(it->red_and_y_feature_map_id));
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&it->green_and_u_feature_map_id), sizeof(it->green_and_u_feature_map_id));
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&it->blue_and_v_feature_map_id), sizeof(it->blue_and_v_feature_map_id));
		}
	}

	float rgb_to_yuv_convert_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count_per_feature_map();

		return static_cast<float>(neuron_count * 9);
	}

	float rgb_to_yuv_convert_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = input_configuration_specific.get_neuron_count_per_feature_map();

		return static_cast<float>(neuron_count * 9);
	}
}
