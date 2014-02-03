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

#include "supervised_transformed_output_data_reader.h"

#include <cstring>

namespace nnforge
{
	supervised_transformed_output_data_reader::supervised_transformed_output_data_reader(
		supervised_data_reader_smart_ptr original_reader,
		data_transformer_smart_ptr transformer)
		: original_reader(original_reader)
		, transformer(transformer)
		, local_output_ptr(0)
		, transformer_sample_count(transformer->get_sample_count())
		, current_sample_id(0)
	{
		if (!transformer->is_in_place())
		{
			buf.resize(original_reader->get_output_configuration().get_neuron_count());
			local_output_ptr = &(*buf.begin());
		}
		if (transformer_sample_count > 1)
		{
			input_buf_size = original_reader->get_input_neuron_elem_size() * original_reader->get_input_configuration().get_neuron_count();
			input_buf.resize(input_buf_size);
			local_input_ptr = &(*input_buf.begin());
		}
	}

	supervised_transformed_output_data_reader::supervised_transformed_output_data_reader()
	{
	}

	supervised_transformed_output_data_reader::~supervised_transformed_output_data_reader()
	{
	}

	bool supervised_transformed_output_data_reader::read(
		void * input_elems,
		float * output_elems)
	{
		bool read = true;
		if (current_sample_id == 0)
		{
			read = original_reader->read(
				(local_input_ptr != 0) && (input_elems != 0) ? local_input_ptr : input_elems,
				(local_output_ptr != 0) && (output_elems != 0) ? local_output_ptr : output_elems);
		}
		if ((local_input_ptr != 0) && (input_elems != 0))
			memcpy(input_elems, local_input_ptr, input_buf_size);

		if (!read)
			return false;

		if (output_elems != 0)
		{
			transformer->transform(
				local_output_ptr,
				output_elems,
				neuron_data_type::type_float,
				original_reader->get_output_configuration(),
				current_sample_id);
		}

		current_sample_id = (current_sample_id + 1) % transformer_sample_count;

		return true;
	}

	void supervised_transformed_output_data_reader::reset()
	{
		current_sample_id = 0;
		transformer->reset();
		original_reader->reset();
	}

	layer_configuration_specific supervised_transformed_output_data_reader::get_input_configuration() const
	{
		return original_reader->get_input_configuration();
	}

	layer_configuration_specific supervised_transformed_output_data_reader::get_output_configuration() const
	{
		return transformer->get_transformed_configuration(original_reader->get_output_configuration());
	}

	unsigned int supervised_transformed_output_data_reader::get_actual_entry_count() const
	{
		return original_reader->get_entry_count() * transformer_sample_count;
	}

	neuron_data_type::input_type supervised_transformed_output_data_reader::get_input_type() const
	{
		return original_reader->get_input_type();
	}

	void supervised_transformed_output_data_reader::set_max_entries_to_read(unsigned int max_entries_to_read)
	{
		original_reader->set_max_entries_to_read(max_entries_to_read);
	}
}
