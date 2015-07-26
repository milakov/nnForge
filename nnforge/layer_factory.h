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

#include "layer.h"

#include <map>
#include <string>
#include <boost/uuid/uuid.hpp>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	class layer_factory
	{
	public:
		bool register_layer(layer::const_ptr sample_layer);

		bool unregister_layer(const std::string& layer_type_name);

		layer::ptr create_layer(const std::string& layer_type_name) const;

	private:
		typedef std::map<std::string, layer::const_ptr> sample_name_map;

		sample_name_map sample_name_layer_map;
	};

	typedef boost::serialization::singleton<layer_factory> single_layer_factory;
}
