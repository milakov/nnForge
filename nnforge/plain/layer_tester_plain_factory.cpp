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

#include "layer_tester_plain_factory.h"
#include "../neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	namespace plain
	{
		bool layer_tester_plain_factory::register_layer_tester_plain(layer_tester_plain_smart_ptr sample_layer_testing_schema)
		{
			return sample_layer_tester_plain_map.insert(sample_map::value_type(sample_layer_testing_schema->get_uuid(), sample_layer_testing_schema)).second;
		}

		bool layer_tester_plain_factory::unregister_layer_tester_plain(const boost::uuids::uuid& layer_guid)
		{
			return sample_layer_tester_plain_map.erase(layer_guid) == 1;
		}

		const_layer_tester_plain_smart_ptr layer_tester_plain_factory::get_tester_plain_layer(const boost::uuids::uuid& layer_guid) const
		{
			sample_map::const_iterator i = sample_layer_tester_plain_map.find(layer_guid);

			if (i == sample_layer_tester_plain_map.end())
				throw neural_network_exception((boost::format("No plain layer tester is registered with id %1%") % layer_guid).str());

			return i->second;
		}
	}
}
