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

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <algorithm>

namespace nnforge
{
	bool layer_factory::register_layer(layer::const_ptr sample_layer)
	{
		std::string type_name = sample_layer->get_type_name();
		std::transform(type_name.begin(), type_name.end(), type_name.begin(), ::tolower);
		return sample_name_layer_map.insert(sample_name_map::value_type(type_name, sample_layer)).second;
	}

	bool layer_factory::unregister_layer(const std::string& layer_type_name)
	{
		return sample_name_layer_map.erase(layer_type_name) == 1;
	}

	layer::ptr layer_factory::create_layer(const std::string& layer_type_name) const
	{
		std::string type_name = layer_type_name;
		std::transform(type_name.begin(), type_name.end(), type_name.begin(), ::tolower);

		sample_name_map::const_iterator i = sample_name_layer_map.find(type_name);

		if (i == sample_name_layer_map.end())
			throw neural_network_exception((boost::format("No layer is registered with name %1%") % layer_type_name).str());

		return i->second->clone();
	}
}
