/*
 *  Copyright 2011-2014 Maxim Milakov
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
#include "data_writer.h"

#include <vector>
#include <ostream>

namespace nnforge
{
	class varying_data_stream_writer : public data_writer
	{
	public:
		// The constructor modifies output_stream to throw exceptions in case of failure
		// The stream should be created with std::ios_base::binary flag
		varying_data_stream_writer(
			nnforge_shared_ptr<std::ostream> output_stream,
			unsigned int entry_count);

		virtual ~varying_data_stream_writer();

		virtual void raw_write(
			const void * all_entry_data,
			size_t data_length);

	private:
		nnforge_shared_ptr<std::ostream> out_stream;

		std::ostream::pos_type start_pos;

		std::vector<unsigned long long> entry_offsets;
		std::ostream::pos_type entry_offsets_pos;
		unsigned int entry_written_count;

	private:
		varying_data_stream_writer(const varying_data_stream_writer&);
		varying_data_stream_writer& operator =(const varying_data_stream_writer&);
	};

	typedef nnforge_shared_ptr<varying_data_stream_writer> varying_data_stream_writer_smart_ptr;
}
