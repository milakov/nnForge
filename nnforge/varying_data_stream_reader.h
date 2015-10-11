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
#include "nn_types.h"

#include <istream>
#include <boost/thread/thread.hpp>

namespace nnforge
{
	class varying_data_stream_reader : public raw_data_reader
	{
	public:
		typedef nnforge_shared_ptr<varying_data_stream_reader> ptr;

		varying_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream);

		virtual ~varying_data_stream_reader();

		// The method returns false in case the entry cannot be read
		virtual bool raw_read(
			unsigned int entry_id,
			std::vector<unsigned char>& all_elems);

		virtual int get_entry_count() const;

		virtual raw_data_writer::ptr get_writer(nnforge_shared_ptr<std::ostream> out) const;

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		std::vector<unsigned long long> entry_offsets;
		std::istream::pos_type reset_pos;
		boost::mutex read_data_from_stream_mutex;
	};
}
