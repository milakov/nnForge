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

#include "structured_data_reader.h"
#include "nn_types.h"

#include <vector>
#include <istream>
#include <boost/thread/thread.hpp>

namespace nnforge
{
	class structured_data_stream_reader : public structured_data_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_data_stream_reader> ptr;

		// The constructor modifies input_stream to throw exceptions in case of failure
		structured_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream);

		virtual ~structured_data_stream_reader();

		virtual bool read(
			unsigned int entry_id,
			float * data);

		virtual layer_configuration_specific get_configuration() const;

		virtual int get_entry_count() const;

		virtual raw_data_writer::ptr get_writer(nnforge_shared_ptr<std::ostream> out) const;

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		unsigned int input_neuron_count;
		layer_configuration_specific input_configuration;
		unsigned int entry_count;
		std::istream::pos_type reset_pos;
		boost::mutex read_data_from_stream_mutex;

	private:
		structured_data_stream_reader(const structured_data_stream_reader&);
		structured_data_stream_reader& operator =(const structured_data_stream_reader&);
	};
}
