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

#pragma once

#include "layer.h"

#include <vector>

// http://en.wikipedia.org/wiki/YUV
namespace nnforge
{
	class color_feature_map_config
	{
	public:
		color_feature_map_config();

		color_feature_map_config(
			unsigned int red_and_y_feature_map_id,
			unsigned int green_and_u_feature_map_id,
			unsigned int blue_and_v_feature_map_id);

		unsigned int red_and_y_feature_map_id;
		unsigned int green_and_u_feature_map_id;
		unsigned int blue_and_v_feature_map_id;
	};

	class rgb_to_yuv_convert_layer : public layer
	{
	public:
		rgb_to_yuv_convert_layer(const std::vector<color_feature_map_config>& color_feature_map_config_list);

		virtual layer_smart_ptr clone() const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_forward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const;

		virtual const boost::uuids::uuid& get_uuid() const;

		virtual void write(std::ostream& binary_stream_to_write_to) const;

		virtual void read(std::istream& binary_stream_to_read_from);

		static const boost::uuids::uuid layer_guid;

	public:
		std::vector<color_feature_map_config> color_feature_map_config_list;
	};
}

