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

#include "rgb_to_yuv_convert_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>

namespace nnforge
{
	const std::string rgb_to_yuv_convert_layer::layer_type_name = "RGBToYUVConvert";

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
		check();
	}

	void rgb_to_yuv_convert_layer::check()
	{
		if (color_feature_map_config_list.empty())
			throw neural_network_exception("Configuration list for RGB to YUV conversion layer may not be empty");
	}

	std::string rgb_to_yuv_convert_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr rgb_to_yuv_convert_layer::clone() const
	{
		return layer::ptr(new rgb_to_yuv_convert_layer(*this));
	}

	layer_configuration_specific rgb_to_yuv_convert_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		for(std::vector<color_feature_map_config>::const_iterator it = color_feature_map_config_list.begin(); it != color_feature_map_config_list.end(); ++it)
		{
			if (it->red_and_y_feature_map_id >= input_configuration_specific_list[0].feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for red and Y (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->red_and_y_feature_map_id % input_configuration_specific_list[0].feature_map_count).str());
			if (it->green_and_u_feature_map_id >= input_configuration_specific_list[0].feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for green and U (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->green_and_u_feature_map_id % input_configuration_specific_list[0].feature_map_count).str());
			if (it->blue_and_v_feature_map_id >= input_configuration_specific_list[0].feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer for blue and V (%1%) is greater or equal than feature map count of input configuration (%2%)") % it->blue_and_v_feature_map_id % input_configuration_specific_list[0].feature_map_count).str());
		}

		return input_configuration_specific_list[0];
	}

	void rgb_to_yuv_convert_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::RGBToYUVConvertParam * param = layer_proto_typed->mutable_rgb_to_yuv_convert_param();
		for(int i = 0; i < color_feature_map_config_list.size(); ++i)
		{
			protobuf::RGBToYUVConvertParam_ColorFeatureMapParam * color_param = param->add_color_feature_map_param();
			color_param->set_red_and_y_feature_map_id(color_feature_map_config_list[i].red_and_y_feature_map_id);
			color_param->set_green_and_u_feature_map_id(color_feature_map_config_list[i].green_and_u_feature_map_id);
			color_param->set_blue_and_v_feature_map_id(color_feature_map_config_list[i].blue_and_v_feature_map_id);
		}
	}

	void rgb_to_yuv_convert_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_rgb_to_yuv_convert_param())
			throw neural_network_exception((boost::format("No rgb_to_yuv_convert_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		color_feature_map_config_list.resize(layer_proto_typed->rgb_to_yuv_convert_param().color_feature_map_param_size());
		for(int i = 0; i < layer_proto_typed->rgb_to_yuv_convert_param().color_feature_map_param_size(); ++i)
		{
			color_feature_map_config_list[i].red_and_y_feature_map_id = layer_proto_typed->rgb_to_yuv_convert_param().color_feature_map_param(i).red_and_y_feature_map_id();
			color_feature_map_config_list[i].green_and_u_feature_map_id = layer_proto_typed->rgb_to_yuv_convert_param().color_feature_map_param(i).green_and_u_feature_map_id();
			color_feature_map_config_list[i].blue_and_v_feature_map_id = layer_proto_typed->rgb_to_yuv_convert_param().color_feature_map_param(i).blue_and_v_feature_map_id();
		}

		check();
	}

	float rgb_to_yuv_convert_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = input_configuration_specific_list[0].get_neuron_count_per_feature_map() * static_cast<unsigned int>(color_feature_map_config_list.size());
				return static_cast<float>(neuron_count * 9);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = input_configuration_specific_list[0].get_neuron_count_per_feature_map() * static_cast<unsigned int>(color_feature_map_config_list.size());
				return static_cast<float>(neuron_count * 9);
			}
		default:
			return 0.0F;
		}
	}
}
