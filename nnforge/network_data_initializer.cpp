/*
 *  Copyright 2011-2016 Maxim Milakov
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
#include "sparse_convolution_layer.h"
#include "rectified_linear_layer.h"
#include "parametric_rectified_linear_layer.h"
#include "add_layer.h"
#include "affine_grid_generator_layer.h"
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
		const network_schema& schema)
	{
		std::vector<layer::const_ptr> layer_list = schema.get_layers();
		for(int i = 0; i < layer_list.size(); ++i)
		{
			float weight_multiplier = 1.0F;
			if (layer_list[i]->get_type_name() == rectified_linear_layer::layer_type_name)
			{
				weight_multiplier *= sqrtf(2.0F);
			}
			if (layer_list[i]->get_type_name() == parametric_rectified_linear_layer::layer_type_name)
			{
				layer_data::ptr data = data_list.find(layer_list[i]->instance_name);
				float a = std::accumulate(data->at(0).begin(), data->at(0).end(), 0.0F) / static_cast<float>(data->at(0).size());
				weight_multiplier *= sqrtf(2.0F / (1.0F + a * a));
			}
			if (layer_list[i]->get_type_name() == add_layer::layer_type_name)
			{
				nnforge_shared_ptr<const add_layer> layer_derived = nnforge_dynamic_pointer_cast<const add_layer>(layer_list[i]);
				weight_multiplier *= 1.0F / std::max(static_cast<int>(layer_list[i]->input_layer_instance_names.size()), 1) / layer_derived->alpha;
			}
			if (layer_list[i]->get_type_name() == affine_grid_generator_layer::layer_type_name)
			{
				weight_multiplier *= 0.01F;
			}

			if ((weight_multiplier != 1.0F) && (!layer_list[i]->input_layer_instance_names.empty()))
			{
				for(std::vector<std::string>::const_iterator it = layer_list[i]->input_layer_instance_names.begin(); it != layer_list[i]->input_layer_instance_names.end(); ++it)
				{
					layer::const_ptr previous_layer = schema.get_layer(*it);
					if ((previous_layer->get_type_name() == convolution_layer::layer_type_name) || (previous_layer->get_type_name() == sparse_convolution_layer::layer_type_name))
					{
						layer_data::ptr data = data_list.find(previous_layer->instance_name);
						std::vector<float>::iterator it_start = data->at(0).begin();
						std::vector<float>::iterator it_end = data->at(0).end();
						for(std::vector<float>::iterator it = it_start; it != it_end; ++it)
							*it *= weight_multiplier;
					}
				}
			}
		}
	}
}
