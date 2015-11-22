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

		virtual layer::ptr clone() const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const;

		virtual float get_flops_per_entry(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_action& action) const;

		virtual std::string get_type_name() const;

		virtual void write_proto(void * layer_proto) const;

		virtual void read_proto(const void * layer_proto);

		static const std::string layer_type_name;

	private:
		void check();

	public:
		std::vector<color_feature_map_config> color_feature_map_config_list;
	};
}

