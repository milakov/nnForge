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

#pragma once

#include "../layer.h"
#include "layer_updater_plain.h"

#include <map>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	namespace plain
	{
		class layer_updater_plain_factory
		{
		public:
			typedef boost::serialization::singleton<layer_updater_plain_factory> singleton;

			bool register_layer_updater_plain(layer_updater_plain::const_ptr sample_layer_updater_plain);

			bool unregister_layer_updater_plain(const std::string& layer_type_name);

			layer_updater_plain::const_ptr get_updater_plain_layer(const std::string& layer_type_name) const;

		private:
			typedef std::map<std::string, layer_updater_plain::const_ptr> sample_map;
			sample_map sample_layer_updater_plain_map;
		};
	}
}
