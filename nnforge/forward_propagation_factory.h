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

#include "forward_propagation.h"
#include "network_schema.h"
#include "debug_state.h"
#include "profile_state.h"

#include <vector>
#include <string>
#include <memory>

namespace nnforge
{
	class forward_propagation_factory
	{
	public:
		typedef std::shared_ptr<forward_propagation_factory> ptr;
		typedef std::shared_ptr<const forward_propagation_factory> const_ptr;

		virtual ~forward_propagation_factory() = default;

		virtual forward_propagation::ptr create(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug,
			profile_state::ptr profile) const = 0;

	protected:
		forward_propagation_factory() = default;
	};
}
