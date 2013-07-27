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

#include "weight_vector_bound_cuda_factory.h"
#include "../neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		bool weight_vector_bound_cuda_factory::register_weight_vector_bound(weight_vector_bound_cuda_smart_ptr sample_weight_vector_bound)
		{
			return sample_weight_vector_bound_map.insert(sample_map::value_type(sample_weight_vector_bound->get_uuid(), sample_weight_vector_bound)).second;
		}

		bool weight_vector_bound_cuda_factory::unregister_weight_vector_bound(const boost::uuids::uuid& layer_guid)
		{
			return sample_weight_vector_bound_map.erase(layer_guid) == 1;
		}

		weight_vector_bound_cuda_smart_ptr weight_vector_bound_cuda_factory::create_weight_vector_bound(
			const_layer_smart_ptr layer,
			cuda_running_configuration_const_smart_ptr cuda_config) const
		{
			sample_map::const_iterator i = sample_weight_vector_bound_map.find(layer->get_uuid());

			if (i == sample_weight_vector_bound_map.end())
				throw neural_network_exception((boost::format("No CUDA weight vector bound is registered with id %1%") % layer->get_uuid()).str());

			return i->second->create(
				layer,
				cuda_config);
		}
	}
}
