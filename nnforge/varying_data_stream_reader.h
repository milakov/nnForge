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

#include <istream>

namespace nnforge
{
	class varying_data_stream_reader : public raw_data_reader
	{
	public:
		varying_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream);

		virtual ~varying_data_stream_reader();

		// The method should return true in case entry is read and false if there is no more entries available (and no entry is read in this case)
		virtual bool raw_read(std::vector<unsigned char>& all_elems);

		virtual void rewind(unsigned int entry_id);

		virtual void reset();

		virtual unsigned int get_entry_count() const;

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		std::vector<unsigned long long> entry_offsets;
		unsigned int entry_read_count;
		std::istream::pos_type reset_pos;
	};

	typedef nnforge_shared_ptr<varying_data_stream_reader> varying_data_stream_reader_smart_ptr;
}
