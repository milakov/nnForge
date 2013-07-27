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
#include "weight_vector_bound_cuda.h"
#include "cuda_running_configuration.h"

#include <map>
#include <vector>
#include <boost/uuid/uuid.hpp>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	namespace cuda
	{
		class weight_vector_bound_cuda_factory
		{
		public:
			bool register_weight_vector_bound(weight_vector_bound_cuda_smart_ptr sample_weight_vector_bound);

			bool unregister_weight_vector_bound(const boost::uuids::uuid& layer_guid);

			weight_vector_bound_cuda_smart_ptr create_weight_vector_bound(
				const_layer_smart_ptr layer,
				cuda_running_configuration_const_smart_ptr cuda_config) const;

		private:
			typedef std::map<boost::uuids::uuid, weight_vector_bound_cuda_smart_ptr> sample_map;
			sample_map sample_weight_vector_bound_map;
		};

		typedef boost::serialization::singleton<weight_vector_bound_cuda_factory> single_weight_vector_bound_factory;
	}
}
