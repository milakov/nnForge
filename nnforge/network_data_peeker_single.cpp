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

#include "network_data_peeker_single.h"

namespace nnforge
{
	network_data_peeker_single::network_data_peeker_single(network_data::ptr data)
		: data(data)
		, is_peeked(false)
	{
	}

	network_data_peeker_single::~network_data_peeker_single()
	{
	}

	network_data_peek_entry network_data_peeker_single::peek(network_schema::ptr schema)
	{
		network_data_peek_entry res;

		if (is_peeked)
			return res;

		is_peeked = true;

		res.index  = 0;
		res.data = data;
		res.start_epoch = 0;

		return res;
	}
}
