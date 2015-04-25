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

#include "layer_data_custom_list.h"

#include "neural_network_exception.h"

namespace nnforge
{
	layer_data_custom_list::layer_data_custom_list()
	{
	}

	layer_data_custom_list::layer_data_custom_list(const const_layer_list& layer_list)
	{
		resize(layer_list.size());
		for(unsigned int i = 0; i < layer_list.size(); ++i)
		{
			at(i) = layer_list[i]->create_layer_data_custom();
		}
	}

	void layer_data_custom_list::check_consistency(const const_layer_list& layer_list) const
	{
		if (size() != layer_list.size())
			throw neural_network_exception("data custom count is not equal layer count");
		for(unsigned int i = 0; i < size(); ++i)
			layer_list[i]->check_layer_data_custom_consistency(*at(i));
	}
}