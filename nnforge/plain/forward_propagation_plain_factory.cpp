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

#include "forward_propagation_plain_factory.h"

#include "forward_propagation_plain.h"

namespace nnforge
{
	namespace plain
	{
		forward_propagation_plain_factory::forward_propagation_plain_factory(plain_running_configuration::const_ptr plain_config)
			: plain_config(plain_config)
		{
		}

		forward_propagation_plain_factory::~forward_propagation_plain_factory()
		{
		}

		forward_propagation::ptr forward_propagation_plain_factory::create(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug) const
		{
			return forward_propagation::ptr(new forward_propagation_plain(schema, output_layer_names, debug, plain_config));
		}
	}
}
