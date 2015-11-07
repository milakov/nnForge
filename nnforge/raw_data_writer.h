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

#include "nn_types.h"

#include <memory>

namespace nnforge
{
	class raw_data_writer
	{
	public:
		typedef nnforge_shared_ptr<raw_data_writer> ptr;

		virtual ~raw_data_writer();

		virtual void raw_write(
			const void * all_entry_data,
			size_t data_length) = 0;

	protected:
		raw_data_writer();

	private:
		raw_data_writer(const raw_data_writer&);
		raw_data_writer& operator =(const raw_data_writer&);
	};
}
