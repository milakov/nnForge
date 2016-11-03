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

#include "structured_data_reader.h"

#include <vector>
#include <istream>
#include <memory>
#include <mutex>

namespace nnforge
{
	class structured_data_stream_reader : public structured_data_reader
	{
	public:
		typedef std::shared_ptr<structured_data_stream_reader> ptr;

		// The constructor modifies input_stream to throw exceptions in case of failure
		structured_data_stream_reader(std::shared_ptr<std::istream> input_stream);

		virtual ~structured_data_stream_reader() = default;

		virtual bool read(
			unsigned int entry_id,
			float * data);

		virtual layer_configuration_specific get_configuration() const;

		virtual int get_entry_count() const;

		virtual raw_data_writer::ptr get_writer(std::shared_ptr<std::ostream> out) const;

	protected:
		std::shared_ptr<std::istream> in_stream;
		unsigned int input_neuron_count;
		layer_configuration_specific input_configuration;
		unsigned int entry_count;
		std::istream::pos_type reset_pos;
		std::mutex read_data_from_stream_mutex;

	private:
		structured_data_stream_reader(const structured_data_stream_reader&) = delete;
		structured_data_stream_reader& operator =(const structured_data_stream_reader&) = delete;
	};
}
