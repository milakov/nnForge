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

#include "../forward_propagation_factory.h"
#include "cuda_running_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class forward_propagation_cuda_factory : public forward_propagation_factory
		{
		public:
			forward_propagation_cuda_factory(cuda_running_configuration::const_ptr cuda_config);

			virtual ~forward_propagation_cuda_factory();

			virtual forward_propagation::ptr create(
				const network_schema& schema,
				const std::vector<std::string>& output_layer_names,
				debug_state::ptr debug) const;

		protected:
			cuda_running_configuration::const_ptr cuda_config;
		};
	}
}
