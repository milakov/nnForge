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

#include "weight_vector_bound_plain_factory.h"

#include "../neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	namespace plain
	{
		bool weight_vector_bound_plain_factory::register_weight_vector_bound_plain(weight_vector_bound_plain_smart_ptr sample_weight_vector_bound_plain)
		{
			return sample_weight_vector_bound_plain_map.insert(sample_map::value_type(sample_weight_vector_bound_plain->get_uuid(), sample_weight_vector_bound_plain)).second;
		}

		bool weight_vector_bound_plain_factory::unregister_weight_vector_bound_plain(const boost::uuids::uuid& layer_guid)
		{
			return sample_weight_vector_bound_plain_map.erase(layer_guid) == 1;
		}

		const_weight_vector_bound_plain_smart_ptr weight_vector_bound_plain_factory::get_updater_plain_layer(const boost::uuids::uuid& layer_guid) const
		{
			sample_map::const_iterator i = sample_weight_vector_bound_plain_map.find(layer_guid);

			if (i == sample_weight_vector_bound_plain_map.end())
				throw neural_network_exception((boost::format("No plain weight vector bound is registered with id %1%") % layer_guid).str());

			return i->second;
		}
	}
}
