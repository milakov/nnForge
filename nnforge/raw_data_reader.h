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
		virtual ~raw_data_reader();

		// The method should return true in case entry is read and false if there is no more entries available (and no entry is read in this case)
		virtual bool raw_read(std::vector<unsigned char>& all_elems) = 0;

		virtual void rewind(unsigned int entry_id) = 0;

		virtual void reset() = 0;

		virtual unsigned int get_entry_count() const = 0;

	protected:
		raw_data_reader();

	private:
		raw_data_reader(const raw_data_reader&);
		raw_data_reader& operator =(const raw_data_reader&);
	};

	typedef nnforge_shared_ptr<raw_data_reader> raw_data_reader_smart_ptr;
}
