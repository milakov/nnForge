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

#include <vector>
#include <memory>

namespace nnforge
{
	class raw_data_reader
	{
	public:
		typedef nnforge_shared_ptr<raw_data_reader> ptr;

		virtual ~raw_data_reader();

		// The method returns false in case the entry cannot be read
		virtual bool raw_read(
			unsigned int entry_id,
			std::vector<unsigned char>& all_elems) = 0;

		// The method should return -1 if entry count is unknown
		virtual int get_entry_count() const = 0;

	protected:
		raw_data_reader();

	private:
		raw_data_reader(const raw_data_reader&);
		raw_data_reader& operator =(const raw_data_reader&);
	};
}
