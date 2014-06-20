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

#include "network_data_initializer.h"

#include "sigmoid_layer.h"
#include "convolution_layer.h"
#include "nn_types.h"

#include <cmath>

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
			if ((layer_list.size() >= 2)
				&& (layer_list[layer_list.size() - 1]->get_uuid() == sigmoid_layer::layer_guid)
				&& (layer_list[layer_list.size() - 2]->get_uuid() == convolution_layer::layer_guid))
			{
				nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_list[layer_list.size() - 2]);
				unsigned int output_feature_map_count = data_list[data_list.size() - 2]->at(1).size();
				if (output_feature_map_count > 1)
				{
					float init_bias = - logf(output_feature_map_count - 1);
					std::fill_n(data_list[data_list.size() - 2]->at(1).begin(), output_feature_map_count, init_bias);
				}
			}
		}
	}
}
