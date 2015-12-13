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

#include "network_schema.h"
#include "network_data.h"
#include "nn_types.h"

namespace nnforge
{
	struct network_data_peek_entry
	{
		unsigned int index;
		network_data::ptr data;
		network_data::ptr momentum_data;
		network_data::ptr momentum_data2;
		unsigned int start_epoch;
	};

	class network_data_peeker
	{
	public:
		typedef nnforge_shared_ptr<network_data_peeker> ptr;

		virtual ~network_data_peeker();

		// The method should return empty data smart pointer in case no more layer data are available
		// The caller is free to modify the data returned
		virtual network_data_peek_entry peek(network_schema::ptr schema) = 0;

	protected:
		network_data_peeker();

	private:
		network_data_peeker(const network_data_peeker&);
		network_data_peeker& operator =(const network_data_peeker&);
	};
}
