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

#pragma once

#include "training_task_state.h"
#include "network_data.h"
#include "nn_types.h"

#include <vector>

namespace nnforge
{
	class network_data_pusher
	{
	public:
		virtual ~network_data_pusher();

		virtual void push(const training_task_state& task_state) = 0;

	protected:
		network_data_pusher();

	private:
		network_data_pusher(const network_data_pusher&);
		network_data_pusher& operator =(const network_data_pusher&);
	};

	typedef nnforge_shared_ptr<network_data_pusher> network_data_pusher_smart_ptr;
}
