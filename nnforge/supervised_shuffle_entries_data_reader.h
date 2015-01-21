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

#include "supervised_data_reader.h"
#include "rnd.h"
#include "nn_types.h"

#include <vector>

namespace nnforge
{
	class supervised_shuffle_entries_data_reader : public supervised_data_reader
	{
	public:
		supervised_shuffle_entries_data_reader(
			supervised_data_reader_smart_ptr original_reader,
			unsigned int block_size);

		virtual ~supervised_shuffle_entries_data_reader();

		// The method should return true in case entry is read and false if there is no more entries available (and no entry is read in this case)
		// If any parameter is null the method should just discard corresponding data
		virtual bool read(
			void * input_elems,
			float * output_elems);

		virtual bool raw_read(std::vector<unsigned char>& all_elems);

		virtual void rewind(unsigned int entry_id);

		virtual void reset();

		virtual void next_epoch();

		virtual layer_configuration_specific get_input_configuration() const;

		virtual layer_configuration_specific get_output_configuration() const;

		virtual neuron_data_type::input_type get_input_type() const;

		virtual unsigned int get_entry_count() const;

	protected:
		supervised_shuffle_entries_data_reader();

		void shuffle();

		void rewind_original();

	protected:
		random_generator generator;

		supervised_data_reader_smart_ptr original_reader;
		unsigned int block_size;
		std::vector<unsigned int> randomized_block_list;

		unsigned int current_block_id;
		unsigned int current_position_in_block;
	};
}
