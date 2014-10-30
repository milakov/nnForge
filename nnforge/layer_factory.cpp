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

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	bool layer_factory::register_layer(layer_smart_ptr sample_layer)
	{
		return sample_layer_map.insert(sample_map::value_type(sample_layer->get_uuid(), sample_layer)).second;
	}

	bool layer_factory::register_layer(const boost::uuids::uuid& layer_guid, layer_smart_ptr sample_layer)
	{
		return sample_layer_map.insert(sample_map::value_type(layer_guid, sample_layer)).second;
	}

	bool layer_factory::unregister_layer(const boost::uuids::uuid& layer_guid)
	{
		return sample_layer_map.erase(layer_guid) == 1;
	}

	layer_smart_ptr layer_factory::create_layer(const boost::uuids::uuid& layer_guid) const
	{
		sample_map::const_iterator i = sample_layer_map.find(layer_guid);

		if (i == sample_layer_map.end())
			throw neural_network_exception((boost::format("No layer is registered with id %1%") % layer_guid).str());

		return i->second->clone();
	}
}
