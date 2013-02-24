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

#include "layer_updater_schema_factory.h"
#include "../neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		bool layer_updater_schema_factory::register_layer_updater_schema(layer_updater_schema_smart_ptr sample_layer_updater_schema)
		{
			return sample_layer_updater_schema_map.insert(sample_map::value_type(sample_layer_updater_schema->get_uuid(), sample_layer_updater_schema)).second;
		}

		bool layer_updater_schema_factory::unregister_layer_updater_schema(const boost::uuids::uuid& layer_guid)
		{
			return sample_layer_updater_schema_map.erase(layer_guid) == 1;
		}

		layer_updater_schema_smart_ptr layer_updater_schema_factory::create_updater_schema_layer(
			const_layer_smart_ptr layer,
			cuda_running_configuration_const_smart_ptr cuda_config) const
		{
			sample_map::const_iterator i = sample_layer_updater_schema_map.find(layer->get_uuid());

			if (i == sample_layer_updater_schema_map.end())
				throw neural_network_exception((boost::format("No CUDA layer updater schema is registered with id %1%") % layer->get_uuid()).str());

			return i->second->create(
				layer,
				cuda_config);
		}
	}
}
