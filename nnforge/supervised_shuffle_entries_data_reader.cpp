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

#include "supervised_shuffle_entries_data_reader.h"

#include <cstring>

namespace nnforge
{
	supervised_shuffle_entries_data_reader::supervised_shuffle_entries_data_reader(
		supervised_data_reader_smart_ptr original_reader,
		unsigned int block_size)
		: original_reader(original_reader)
		, block_size(block_size)
		, current_block_id(0)
		, current_position_in_block(0)
		, generator(rnd::get_random_generator(64321019))
	{
		unsigned int block_count = (original_reader->get_entry_count() + block_size - 1) / block_size;
		randomized_block_list.resize(block_count);
		shuffle();
		rewind_original();
	}

	supervised_shuffle_entries_data_reader::supervised_shuffle_entries_data_reader()
	{
	}

	supervised_shuffle_entries_data_reader::~supervised_shuffle_entries_data_reader()
	{
	}

	bool supervised_shuffle_entries_data_reader::read(
		void * input_elems,
		float * output_elems)
	{
		bool entry_read = original_reader->read(input_elems, output_elems);
		if (entry_read)
		{
			++current_position_in_block;
			if (current_position_in_block >= block_size)
			{
				current_position_in_block = 0;
				++current_block_id;
				rewind_original();
			}
		}

		return entry_read;
	}

	void supervised_shuffle_entries_data_reader::reset()
	{
		rewind(0);
	}

	void supervised_shuffle_entries_data_reader::next_epoch()
	{
		original_reader->next_epoch();
		shuffle();
		rewind(0);
	}

	layer_configuration_specific supervised_shuffle_entries_data_reader::get_input_configuration() const
	{
		return original_reader->get_input_configuration();
	}

	layer_configuration_specific supervised_shuffle_entries_data_reader::get_output_configuration() const
	{
		return original_reader->get_output_configuration();
	}

	unsigned int supervised_shuffle_entries_data_reader::get_entry_count() const
	{
		return original_reader->get_entry_count();
	}

	neuron_data_type::input_type supervised_shuffle_entries_data_reader::get_input_type() const
	{
		return original_reader->get_input_type();
	}

	void supervised_shuffle_entries_data_reader::rewind(unsigned int entry_id)
	{
		current_block_id = entry_id / block_size;
		current_position_in_block = entry_id - current_block_id * block_size;
		rewind_original();
	}

	void supervised_shuffle_entries_data_reader::rewind_original()
	{
		if (current_block_id < randomized_block_list.size())
			original_reader->rewind(randomized_block_list[current_block_id] * block_size + current_position_in_block);
	}

	bool supervised_shuffle_entries_data_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		bool entry_read = original_reader->raw_read(all_elems);
		if (entry_read)
		{
			++current_position_in_block;
			if (current_position_in_block >= block_size)
			{
				current_position_in_block = 0;
				++current_block_id;
				rewind_original();
			}
		}

		return entry_read;
	}

	void supervised_shuffle_entries_data_reader::shuffle()
	{
		for(int i = 0; i < randomized_block_list.size(); ++i)
			randomized_block_list[i] = i;

		// The last block remains the last one
		for(int i = static_cast<int>(randomized_block_list.size()) - 2; i > 0; --i)
		{
			nnforge_uniform_int_distribution<unsigned int> dist(0, i);
			std::swap(randomized_block_list[i], randomized_block_list[dist(generator)]);
		}
	}
}
