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

#pragma once

#include "backward_propagation.h"
#include "nn_types.h"
#include "network_schema.h"
#include "debug_state.h"
#include "profile_state.h"

#include <vector>
#include <string>

namespace nnforge
{
	class backward_propagation_factory
	{
	public:
		typedef nnforge_shared_ptr<backward_propagation_factory> ptr;
		typedef nnforge_shared_ptr<const backward_propagation_factory> const_ptr;

		virtual ~backward_propagation_factory();

		virtual backward_propagation::ptr create(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names,
			debug_state::ptr debug,
			profile_state::ptr profile) const = 0;

	protected:
		backward_propagation_factory();
	};
}
