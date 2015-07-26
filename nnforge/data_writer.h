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

#include "raw_data_reader.h"

namespace nnforge
{
	class data_writer
	{
	public:
		typedef nnforge_shared_ptr<data_writer> ptr;

		virtual ~data_writer();

		virtual void raw_write(
			const void * all_elems,
			size_t data_length) = 0;

		void write_randomized(raw_data_reader& reader);

	protected:
		data_writer();
	};
}
