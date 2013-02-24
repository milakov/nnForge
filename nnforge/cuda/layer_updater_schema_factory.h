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

#pragma once

#include "../layer.h"
#include "layer_updater_schema.h"
#include "cuda_running_configuration.h"

#include <map>
#include <vector>
#include <boost/uuid/uuid.hpp>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	namespace cuda
	{
		class layer_updater_schema_factory
		{
		public:
			bool register_layer_updater_schema(layer_updater_schema_smart_ptr sample_layer_updater_schema);

			bool unregister_layer_updater_schema(const boost::uuids::uuid& layer_guid);

			layer_updater_schema_smart_ptr create_updater_schema_layer(
				const_layer_smart_ptr layer,
				cuda_running_configuration_const_smart_ptr cuda_config) const;

		private:
			typedef std::map<boost::uuids::uuid, layer_updater_schema_smart_ptr> sample_map;
			sample_map sample_layer_updater_schema_map;
		};

		typedef boost::serialization::singleton<layer_updater_schema_factory> single_layer_updater_schema_factory;
	}
}
