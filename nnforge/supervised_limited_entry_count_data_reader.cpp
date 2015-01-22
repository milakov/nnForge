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

#include "supervised_limited_entry_count_data_reader.h"

#include <cstring>

namespace nnforge
{
	supervised_limited_entry_count_data_reader::supervised_limited_entry_count_data_reader(
		supervised_data_reader_smart_ptr original_reader,
		unsigned int max_entry_count)
		: original_reader(original_reader)
		, max_entry_count(max_entry_count)
		, entry_read_count(0)
	{
	}

	supervised_limited_entry_count_data_reader::supervised_limited_entry_count_data_reader()
	{
	}

	supervised_limited_entry_count_data_reader::~supervised_limited_entry_count_data_reader()
	{
	}

	bool supervised_limited_entry_count_data_reader::entry_available()
	{
		return (entry_read_count < std::min(max_entry_count, original_reader->get_entry_count()));
	}

	bool supervised_limited_entry_count_data_reader::read(
		void * input_elems,
		float * output_elems)
	{
		if (!entry_available())
			return false;

		++entry_read_count;
		return original_reader->read(input_elems, output_elems);
	}

	void supervised_limited_entry_count_data_reader::reset()
	{
		entry_read_count = 0;
		original_reader->reset();
	}

	layer_configuration_specific supervised_limited_entry_count_data_reader::get_input_configuration() const
	{
		return original_reader->get_input_configuration();
	}

	layer_configuration_specific supervised_limited_entry_count_data_reader::get_output_configuration() const
	{
		return original_reader->get_output_configuration();
	}

	unsigned int supervised_limited_entry_count_data_reader::get_entry_count() const
	{
		return std::min(max_entry_count, original_reader->get_entry_count());
	}

	neuron_data_type::input_type supervised_limited_entry_count_data_reader::get_input_type() const
	{
		return original_reader->get_input_type();
	}

	void supervised_limited_entry_count_data_reader::rewind(unsigned int entry_id)
	{
		original_reader->rewind(entry_id);
	}

	bool supervised_limited_entry_count_data_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		if (!entry_available())
			return false;

		++entry_read_count;
		return original_reader->raw_read(all_elems);
	}

	void supervised_limited_entry_count_data_reader::next_epoch()
	{
		entry_read_count = 0;
		original_reader->next_epoch();
	}

	unsigned int supervised_limited_entry_count_data_reader::get_sample_count() const
	{
		return original_reader->get_sample_count();
	}
}
