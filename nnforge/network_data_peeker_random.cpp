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

#include "network_data_peeker_random.h"

namespace nnforge
{
	network_data_peeker_random::network_data_peeker_random(
		unsigned int max_network_data_count,
		unsigned int base_index)
		: max_network_data_count(max_network_data_count)
		, base_index(base_index)
		, generated_network_data_count(0)
		, gen(rnd::get_random_generator())
	{
	}

	network_data_peeker_random::~network_data_peeker_random()
	{
	}

	std::pair<unsigned int, network_data_smart_ptr> network_data_peeker_random::peek(network_schema_smart_ptr schema)
	{
		if (generated_network_data_count >= max_network_data_count)
			return std::make_pair(base_index, network_data_smart_ptr());

		network_data_smart_ptr data(new network_data(*schema));

		data->randomize(
			*schema,
			gen);

		std::pair<unsigned int, network_data_smart_ptr> res(generated_network_data_count + base_index, data);

		++generated_network_data_count;

		return res;
	}
}
