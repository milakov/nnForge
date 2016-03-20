/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "layer_configuration_specific.h"
#include "nn_types.h"
#include "structured_data_writer.h"

#include <vector>
#include <ostream>

namespace nnforge
{
	class structured_data_stream_writer : public structured_data_writer
	{
	public:
		typedef nnforge_shared_ptr<structured_data_stream_writer> ptr;

		// The constructor modifies output_stream to throw exceptions in case of failure
		// The stream should be created with std::ios_base::binary flag
		structured_data_stream_writer(
			nnforge_shared_ptr<std::ostream> output_stream,
			const layer_configuration_specific& config);

		virtual ~structured_data_stream_writer();

		virtual void write(const float * neurons);

		virtual void write(
			unsigned int entry_id,
			const float * neurons);

	private:
		nnforge_shared_ptr<std::ostream> out_stream;

		unsigned int neuron_count;
		std::ostream::pos_type entry_count_pos;
		unsigned int entry_count;

	private:
		structured_data_stream_writer(const structured_data_stream_writer&);
		structured_data_stream_writer& operator =(const structured_data_stream_writer&);
	};
}
