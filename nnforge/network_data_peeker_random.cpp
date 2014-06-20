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
		network_output_type::output_type network_type,
		unsigned int max_network_data_count,
		unsigned int base_index)
		: network_type(network_type)
		, max_network_data_count(max_network_data_count)
		, base_index(base_index)
		, generated_network_data_count(0)
		, gen(rnd::get_random_generator())
	{
	}

	network_data_peeker_random::~network_data_peeker_random()
	{
	}

	network_data_peek_entry network_data_peeker_random::peek(network_schema_smart_ptr schema)
	{
		network_data_peek_entry res;

		if (generated_network_data_count >= max_network_data_count)
			return res;

		network_data_smart_ptr data(new network_data(*schema));

		data->randomize(
			*schema,
			gen);
		init.initialize(
			*data,
			*schema,
			network_type);

		res.index = generated_network_data_count + base_index;
		res.data = data;
		res.start_epoch = 0;

		++generated_network_data_count;

		return res;
	}
}
