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

#include "network_data_initializer.h"

#include "sigmoid_layer.h"
#include "hyperbolic_tangent_layer.h"
#include "convolution_layer.h"
#include "rectified_linear_layer.h"
#include "parametric_rectified_linear_layer.h"
#include "nn_types.h"

#include <cmath>
#include <numeric>

namespace nnforge
{
	network_data_initializer::network_data_initializer()
	{
	}

	void network_data_initializer::initialize(
		layer_data_list& data_list,
		const const_layer_list& layer_list,
		network_output_type::output_type type)
	{
		if (type == network_output_type::type_classifier)
		{
			if (layer_list.size() >= 2)
			{
				if ((layer_list[layer_list.size() - 1]->get_uuid() == sigmoid_layer::layer_guid)
					&& (layer_list[layer_list.size() - 2]->get_uuid() == convolution_layer::layer_guid))
				{
					unsigned int output_feature_map_count = static_cast<unsigned int>(data_list[data_list.size() - 2]->at(1).size());
					if (output_feature_map_count > 1)
					{
						float init_bias = - logf(static_cast<float>(output_feature_map_count - 1));
						std::fill_n(data_list[data_list.size() - 2]->at(1).begin(), output_feature_map_count, init_bias);
					}
				}
				else if ((layer_list[layer_list.size() - 1]->get_uuid() == hyperbolic_tangent_layer::layer_guid)
					&& (layer_list[layer_list.size() - 2]->get_uuid() == convolution_layer::layer_guid))
				{
					unsigned int output_feature_map_count = static_cast<unsigned int>(data_list[data_list.size() - 2]->at(1).size());
					if (output_feature_map_count > 1)
					{
						float init_bias = -1.0F + (2.0F / output_feature_map_count);
						std::fill_n(data_list[data_list.size() - 2]->at(1).begin(), output_feature_map_count, init_bias);
					}
				}
			}
		}

		float weight_multiplier = 1.0F;
		for(int i = 0; i < layer_list.size(); ++i)
		{
			if (layer_list[i]->get_uuid() == rectified_linear_layer::layer_guid)
			{
				weight_multiplier *= sqrtf(2.0F);
			}
			if (layer_list[i]->get_uuid() == parametric_rectified_linear_layer::layer_guid)
			{
				float a = std::accumulate(data_list[i]->at(0).begin(), data_list[i]->at(0).end(), 0.0F) / static_cast<float>(data_list[i]->at(0).size());
				weight_multiplier *= sqrtf(2.0F / (1.0F + a * a));
			}
			else if ((layer_list[i]->get_uuid() == convolution_layer::layer_guid) && (weight_multiplier != 1.0F))
			{
				std::vector<float>::iterator it_start = data_list[i]->at(0).begin();
				std::vector<float>::iterator it_end = data_list[i]->at(0).end();
				for(std::vector<float>::iterator it = it_start; it != it_end; ++it)
					*it *= weight_multiplier;

				weight_multiplier = 1.0F;
			}
		}
	}
}
