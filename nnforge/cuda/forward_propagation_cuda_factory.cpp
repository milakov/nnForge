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

#include "forward_propagation_cuda_factory.h"

#include "forward_propagation_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		forward_propagation_cuda_factory::forward_propagation_cuda_factory(cuda_running_configuration::const_ptr cuda_config)
			: cuda_config(cuda_config)
		{
		}

		forward_propagation::ptr forward_propagation_cuda_factory::create(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug,
			profile_state::ptr profile) const
		{
			return forward_propagation::ptr(new forward_propagation_cuda(schema, output_layer_names, debug, profile, cuda_config));
		}
	}
}
