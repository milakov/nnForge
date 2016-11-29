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

#pragma once

#include "../layer.h"
#include "layer_testing_schema.h"
#include "cuda_running_configuration.h"

#include <map>
#include <vector>

namespace nnforge
{
	namespace cuda
	{
		class layer_testing_schema_factory
		{
		public:
			bool register_layer_testing_schema(layer_testing_schema::const_ptr sample_layer_testing_schema);

			bool unregister_layer_testing_schema(const std::string& layer_type_name);

			layer_testing_schema::ptr create_testing_schema_layer(layer::const_ptr layer) const;

			static layer_testing_schema_factory& get_singleton();

		private:
			layer_testing_schema_factory() = default;

			typedef std::map<std::string, layer_testing_schema::const_ptr> sample_map;
			sample_map sample_layer_testing_schema_map;
		};
	}
}
