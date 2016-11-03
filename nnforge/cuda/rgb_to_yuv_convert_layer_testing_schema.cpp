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

#include "rgb_to_yuv_convert_layer_testing_schema.h"

#include "../neural_network_exception.h"
#include "../rgb_to_yuv_convert_layer.h"
#include "rgb_to_yuv_convert_layer_tester_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		std::string rgb_to_yuv_convert_layer_testing_schema::get_type_name() const
		{
			return rgb_to_yuv_convert_layer::layer_type_name;
		}

		layer_testing_schema::ptr rgb_to_yuv_convert_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema::ptr(new rgb_to_yuv_convert_layer_testing_schema());
		}

		layer_tester_cuda::ptr rgb_to_yuv_convert_layer_testing_schema::create_tester_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_tester_cuda::ptr(new rgb_to_yuv_convert_layer_tester_cuda());
		}

		std::vector<cuda_linear_buffer_device::const_ptr> rgb_to_yuv_convert_layer_testing_schema::get_schema_buffers() const
		{
			std::vector<cuda_linear_buffer_device::const_ptr> res;

			std::shared_ptr<const rgb_to_yuv_convert_layer> layer_derived = std::dynamic_pointer_cast<const rgb_to_yuv_convert_layer>(layer_schema);
			std::vector<int> color_feature_map_config_raw_value_list;
			for(std::vector<color_feature_map_config>::const_iterator it = layer_derived->color_feature_map_config_list.begin(); it != layer_derived->color_feature_map_config_list.end(); ++it)
			{
				color_feature_map_config_raw_value_list.push_back(it->red_and_y_feature_map_id);
				color_feature_map_config_raw_value_list.push_back(it->green_and_u_feature_map_id);
				color_feature_map_config_raw_value_list.push_back(it->blue_and_v_feature_map_id);
			}

			res.push_back(
				cuda_linear_buffer_device::const_ptr(new cuda_linear_buffer_device(
					&(*color_feature_map_config_raw_value_list.begin()),
					color_feature_map_config_raw_value_list.size() * sizeof(int)))
				);

			return res;
		}
	}
}
