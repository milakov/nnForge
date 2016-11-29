/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "layer_testing_schema_factory.h"
#include "../neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		bool layer_testing_schema_factory::register_layer_testing_schema(layer_testing_schema::const_ptr sample_layer_testing_schema)
		{
			return sample_layer_testing_schema_map.insert(sample_map::value_type(sample_layer_testing_schema->get_type_name(), sample_layer_testing_schema)).second;
		}

		bool layer_testing_schema_factory::unregister_layer_testing_schema(const std::string& layer_name)
		{
			return sample_layer_testing_schema_map.erase(layer_name) == 1;
		}

		layer_testing_schema::ptr layer_testing_schema_factory::create_testing_schema_layer(layer::const_ptr layer) const
		{
			sample_map::const_iterator i = sample_layer_testing_schema_map.find(layer->get_type_name());

			if (i == sample_layer_testing_schema_map.end())
				throw neural_network_exception((boost::format("No CUDA layer testing schema is registered with type name %1%") % layer->get_type_name()).str());

			return i->second->create(layer);
		}

		layer_testing_schema_factory& layer_testing_schema_factory::get_singleton()
		{
			static layer_testing_schema_factory instance;
			return instance;
		}
	}
}
