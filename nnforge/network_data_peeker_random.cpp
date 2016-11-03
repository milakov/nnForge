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

#include "network_data_peeker_random.h"

namespace nnforge
{
	network_data_peeker_random::network_data_peeker_random(
		unsigned int max_network_data_count,
		unsigned int base_index,
		const std::vector<network_data_peek_entry>& leading_tasks)
		: max_network_data_count(max_network_data_count)
		, base_index(base_index)
		, leading_tasks(leading_tasks)
		, trained_network_data_count(0)
		, generated_network_data_count(0)
		, gen(rnd::get_random_generator())
	{
	}

	network_data_peek_entry network_data_peeker_random::peek(network_schema::ptr schema)
	{
		network_data_peek_entry res;

		if (trained_network_data_count >= max_network_data_count)
			return res;

		if (leading_tasks.empty())
		{
			network_data::ptr data(new network_data(schema->get_layers()));

			data->randomize(
				schema->get_layers(),
				gen);
			init.initialize(
				data->data_list,
				*schema);

			res.index = generated_network_data_count + base_index;
			res.data = data;
			res.start_epoch = 0;

			++generated_network_data_count;
		}
		else
		{
			res = leading_tasks.back();

			leading_tasks.pop_back();
		}

		++trained_network_data_count;

		return res;
	}
}
