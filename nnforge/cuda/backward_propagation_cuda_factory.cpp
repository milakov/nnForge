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

#include "backward_propagation_cuda_factory.h"

#include "backward_propagation_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		backward_propagation_cuda_factory::backward_propagation_cuda_factory(cuda_running_configuration::const_ptr cuda_config)
			: cuda_config(cuda_config)
		{
		}

		backward_propagation_cuda_factory::~backward_propagation_cuda_factory()
		{
		}

		backward_propagation::ptr backward_propagation_cuda_factory::create(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names,
			debug_state::ptr debug,
			profile_state::ptr profile) const
		{
			return backward_propagation::ptr(new backward_propagation_cuda(
				schema,
				output_layer_names,
				error_source_layer_names,
				exclude_data_update_layer_names,
				debug,
				profile,
				cuda_config));
		}
	}
}
