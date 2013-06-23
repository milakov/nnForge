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

#include "supervised_transformed_data_reader.h"

namespace nnforge{
	supervised_transformed_data_reader::supervised_transformed_data_reader(
		supervised_data_reader_smart_ptr original_reader,
		data_transformer_smart_ptr transformer)
		: original_reader(original_reader)
		, transformer(transformer)
		, local_input_ptr(0)
	{
		if (!transformer->is_in_place())
		{
			buf.resize(original_reader->get_input_neuron_elem_size() * original_reader->get_input_configuration().get_neuron_count());
			local_input_ptr = &(*buf.begin());
		}
	}

	supervised_transformed_data_reader::supervised_transformed_data_reader()
	{
	}

	supervised_transformed_data_reader::~supervised_transformed_data_reader()
	{
	}

	bool supervised_transformed_data_reader::read(
		void * input_elems,
		float * output_elems)
	{
		bool read = original_reader->read(local_input_ptr ? local_input_ptr : input_elems, output_elems);

		if (!read)
			return false;

		if (input_elems != 0)
		{
			transformer->transform(
				local_input_ptr,
				input_elems,
				original_reader->get_input_type(),
				original_reader->get_input_configuration());
		}

		return true;
	}

	void supervised_transformed_data_reader::reset()
	{
		transformer->reset();
		original_reader->reset();
	}

	layer_configuration_specific supervised_transformed_data_reader::get_input_configuration() const
	{
		return transformer->get_transformed_configuration(original_reader->get_input_configuration());
	}

	layer_configuration_specific supervised_transformed_data_reader::get_output_configuration() const
	{
		return original_reader->get_output_configuration();
	}

	unsigned int supervised_transformed_data_reader::get_entry_count() const
	{
		return original_reader->get_entry_count();
	}

	neuron_data_type::input_type supervised_transformed_data_reader::get_input_type() const
	{
		return original_reader->get_input_type();
	}
}
