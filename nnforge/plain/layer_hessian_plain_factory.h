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
#include "layer_hessian_plain.h"
#include "plain_running_configuration.h"

#include <map>
#include <vector>
#include <boost/uuid/uuid.hpp>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	namespace plain
	{
		class layer_hessian_plain_factory
		{
		public:
			bool register_layer_hessian_plain(layer_hessian_plain_smart_ptr sample_layer_hessian_plain);

			bool unregister_layer_hessian_plain(const boost::uuids::uuid& layer_guid);

			const_layer_hessian_plain_smart_ptr get_hessian_plain_layer(const boost::uuids::uuid& layer_guid) const;

		private:
			typedef std::map<boost::uuids::uuid, layer_hessian_plain_smart_ptr> sample_map;
			sample_map sample_layer_hessian_plain_map;
		};

		typedef boost::serialization::singleton<layer_hessian_plain_factory> single_layer_hessian_plain_factory;
	}
}
