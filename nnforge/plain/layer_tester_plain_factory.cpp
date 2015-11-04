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

#include "layer_tester_plain_factory.h"

#include "../neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace plain
	{
		bool layer_tester_plain_factory::register_layer_tester_plain(layer_tester_plain::const_ptr sample_layer_tester_plain)
		{
			return sample_layer_tester_plain_map.insert(sample_map::value_type(sample_layer_tester_plain->get_type_name(), sample_layer_tester_plain)).second;
		}

		bool layer_tester_plain_factory::unregister_layer_tester_plain(const std::string& layer_type_name)
		{
			return sample_layer_tester_plain_map.erase(layer_type_name) == 1;
		}

		layer_tester_plain::const_ptr layer_tester_plain_factory::get_tester_plain_layer(const std::string& layer_type_name) const
		{
			sample_map::const_iterator i = sample_layer_tester_plain_map.find(layer_type_name);

			if (i == sample_layer_tester_plain_map.end())
				throw neural_network_exception((boost::format("No plain layer tester is registered with type %1%") % layer_type_name).str());

			return i->second;
		}
	}
}
