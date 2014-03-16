/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "rgb_to_yuv_convert_layer_updater_schema.h"

#include "../rgb_to_yuv_convert_layer.h"
#include "rgb_to_yuv_convert_layer_updater_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		rgb_to_yuv_convert_layer_updater_schema::rgb_to_yuv_convert_layer_updater_schema()
		{
		}

		rgb_to_yuv_convert_layer_updater_schema::~rgb_to_yuv_convert_layer_updater_schema()
		{
		}

		std::tr1::shared_ptr<layer_updater_schema> rgb_to_yuv_convert_layer_updater_schema::create_specific() const
		{
			return layer_updater_schema_smart_ptr(new rgb_to_yuv_convert_layer_updater_schema());
		}

		const boost::uuids::uuid& rgb_to_yuv_convert_layer_updater_schema::get_uuid() const
		{
			return rgb_to_yuv_convert_layer::layer_guid;
		}

		layer_updater_cuda_smart_ptr rgb_to_yuv_convert_layer_updater_schema::create_updater_specific(
			const layer_configuration_specific& hyperbolic_tangent_layer_hessian_schema,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_updater_cuda_smart_ptr(new rgb_to_yuv_convert_layer_updater_cuda());
		}

		std::vector<const_cuda_linear_buffer_device_smart_ptr> rgb_to_yuv_convert_layer_updater_schema::get_schema_buffers() const
		{
			std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

			std::tr1::shared_ptr<const rgb_to_yuv_convert_layer> layer_derived = std::tr1::dynamic_pointer_cast<const rgb_to_yuv_convert_layer>(layer_schema);
			std::vector<int> color_feature_map_config_raw_value_list;
			for(std::vector<color_feature_map_config>::const_iterator it = layer_derived->color_feature_map_config_list.begin(); it != layer_derived->color_feature_map_config_list.end(); ++it)
			{
				color_feature_map_config_raw_value_list.push_back(it->red_and_y_feature_map_id);
				color_feature_map_config_raw_value_list.push_back(it->green_and_u_feature_map_id);
				color_feature_map_config_raw_value_list.push_back(it->blue_and_v_feature_map_id);
			}

			res.push_back(
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*color_feature_map_config_raw_value_list.begin()),
					color_feature_map_config_raw_value_list.size() * sizeof(int)))
				);

			return res;
		}
	}
}
