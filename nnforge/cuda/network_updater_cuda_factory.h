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

#include "../network_updater_factory.h"
#include "cuda_running_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class network_updater_cuda_factory : public network_updater_factory
		{
		public:
			network_updater_cuda_factory(cuda_running_configuration_const_smart_ptr cuda_config);

			virtual ~network_updater_cuda_factory();

			virtual network_updater_smart_ptr create(
				network_schema_smart_ptr schema,
				const_error_function_smart_ptr ef,
				const std::map<unsigned int, float>& layer_to_dropout_rate_map,
				const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
				float weight_decay) const;

		protected:
			cuda_running_configuration_const_smart_ptr cuda_config;
		};
	}
}
