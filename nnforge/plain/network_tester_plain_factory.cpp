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

#include "network_tester_plain_factory.h"

#include "network_tester_plain.h"

namespace nnforge
{
	namespace plain
	{
		network_tester_plain_factory::network_tester_plain_factory(plain_running_configuration_const_smart_ptr plain_config)
			: plain_config(plain_config)
		{
		}

		network_tester_plain_factory::~network_tester_plain_factory()
		{
		}

		network_tester_smart_ptr network_tester_plain_factory::create(
			network_schema_smart_ptr schema,
			const_data_scale_params_smart_ptr scale_params) const
		{
			return network_tester_smart_ptr(new network_tester_plain(schema, scale_params, plain_config));
		}
	}
}
