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

#include "unsupervised_transformed_input_data_reader.h"

namespace nnforge
{
	unsupervised_transformed_input_data_reader::unsupervised_transformed_input_data_reader(
		unsupervised_data_reader_smart_ptr original_reader,
		data_transformer_smart_ptr transformer)
		: original_reader(original_reader)
		, transformer(transformer)
		, local_input_ptr(0)
		, transformer_sample_count(transformer->get_sample_count())
		, current_sample_id(0)
	{
		if (!transformer->is_in_place())
		{
			buf.resize(neuron_data_type::get_input_size(transformer->get_transformed_data_type(original_reader->get_input_type()))
				* original_reader->get_input_configuration().get_neuron_count());
			local_input_ptr = &(*buf.begin());
		}
	}

	unsupervised_transformed_input_data_reader::unsupervised_transformed_input_data_reader()
	{
	}

	unsupervised_transformed_input_data_reader::~unsupervised_transformed_input_data_reader()
	{
	}

	bool unsupervised_transformed_input_data_reader::read(void * input_elems)
	{
		bool read = true;
		if (current_sample_id == 0)
			read = original_reader->read((local_input_ptr != 0) && (input_elems != 0) ? local_input_ptr : input_elems);

		if (!read)
			return false;

		if (input_elems != 0)
		{
			transformer->transform(
				local_input_ptr,
				input_elems,
				original_reader->get_input_type(),
				original_reader->get_input_configuration(),
				current_sample_id);
		}

		current_sample_id = (current_sample_id + 1) % transformer_sample_count;

		return true;
	}

	void unsupervised_transformed_input_data_reader::reset()
	{
		current_sample_id = 0;
		transformer->reset();
		original_reader->reset();
	}

	void unsupervised_transformed_input_data_reader::next_epoch()
	{
		current_sample_id = 0;
		transformer->reset();
		original_reader->next_epoch();
	}

	layer_configuration_specific unsupervised_transformed_input_data_reader::get_input_configuration() const
	{
		return transformer->get_transformed_configuration(original_reader->get_input_configuration());
	}

	unsigned int unsupervised_transformed_input_data_reader::get_entry_count() const
	{
		return original_reader->get_entry_count() * transformer_sample_count;
	}

	neuron_data_type::input_type unsupervised_transformed_input_data_reader::get_input_type() const
	{
		return transformer->get_transformed_data_type(original_reader->get_input_type());
	}

	void unsupervised_transformed_input_data_reader::rewind(unsigned int entry_id)
	{
		throw std::runtime_error("rewind not implemented for unsupervised_transformed_input_data_reader");
	}

	bool unsupervised_transformed_input_data_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		throw std::runtime_error("raw_read not implemented for unsupervised_transformed_input_data_reader");
	}

	unsigned int unsupervised_transformed_input_data_reader::get_sample_count() const
	{
		return transformer_sample_count * original_reader->get_sample_count();
	}
}
