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
		const std::vector<layer::const_ptr>& layer_list,
		network_output_type::output_type type)
	{
		// FIXME!!! assuming sequential layers
		if (type == network_output_type::type_classifier)
		{
			if (layer_list.size() >= 2)
			{
				if ((layer_list[layer_list.size() - 1]->get_type_name() == sigmoid_layer::layer_type_name)
					&& (layer_list[layer_list.size() - 2]->get_type_name() == convolution_layer::layer_type_name))
				{
					layer_data::ptr data = data_list.find(layer_list[layer_list.size() - 2]->instance_name);
					unsigned int output_feature_map_count = static_cast<unsigned int>(data->at(1).size());
					if (output_feature_map_count > 1)
					{
						float init_bias = - logf(static_cast<float>(output_feature_map_count - 1));
						std::fill_n(data->at(1).begin(), output_feature_map_count, init_bias);
					}
				}
			}
		}

		float weight_multiplier = 1.0F;
		for(int i = 0; i < layer_list.size(); ++i)
		{
			if (layer_list[i]->get_type_name() == rectified_linear_layer::layer_type_name)
			{
				weight_multiplier *= sqrtf(2.0F);
			}
			if (layer_list[i]->get_type_name() == parametric_rectified_linear_layer::layer_type_name)
			{
				layer_data::ptr data = data_list.find(layer_list[layer_list.size() - 2]->instance_name);
				float a = std::accumulate(data->at(0).begin(), data->at(0).end(), 0.0F) / static_cast<float>(data->at(0).size());
				weight_multiplier *= sqrtf(2.0F / (1.0F + a * a));
			}
			else if ((layer_list[i]->get_type_name() == convolution_layer::layer_type_name) && (weight_multiplier != 1.0F))
			{
				layer_data::ptr data = data_list.find(layer_list[layer_list.size() - 2]->instance_name);
				std::vector<float>::iterator it_start = data->at(0).begin();
				std::vector<float>::iterator it_end = data->at(0).end();
				for(std::vector<float>::iterator it = it_start; it != it_end; ++it)
					*it *= weight_multiplier;

				weight_multiplier = 1.0F;
			}
		}
	}
}
