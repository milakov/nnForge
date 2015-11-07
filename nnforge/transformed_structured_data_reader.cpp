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

#include "transformed_structured_data_reader.h"

namespace nnforge
{
	transformed_structured_data_reader::transformed_structured_data_reader(
		structured_data_reader::ptr original_reader,
		data_transformer::ptr transformer)
		: original_reader(original_reader)
		, transformer(transformer)
		, transformer_sample_count(transformer->get_sample_count())
		, original_config(original_reader->get_configuration())
	{
	}

	transformed_structured_data_reader::transformed_structured_data_reader()
	{
	}

	transformed_structured_data_reader::~transformed_structured_data_reader()
	{
	}

	bool transformed_structured_data_reader::read(
		unsigned int entry_id,
		float * data)
	{
		std::vector<float> original_data(original_config.get_neuron_count());

		if (!original_reader->read(entry_id / transformer_sample_count, &original_data[0]))
			return false;

		transformer->transform(
			&original_data[0],
			data,
			original_config,
			entry_id % transformer_sample_count);

		return true;
	}

	layer_configuration_specific transformed_structured_data_reader::get_configuration() const
	{
		return transformer->get_transformed_configuration(original_config);
	}

	int transformed_structured_data_reader::get_entry_count() const
	{
		return original_reader->get_entry_count() * transformer_sample_count;
	}

	bool transformed_structured_data_reader::raw_read(
		unsigned int entry_id,
		std::vector<unsigned char>& all_elems)
	{
		throw std::runtime_error("raw_read not implemented for transformed_structured_data_reader");
	}

	raw_data_writer::ptr transformed_structured_data_reader::get_writer(nnforge_shared_ptr<std::ostream> out) const
	{
		throw std::runtime_error("get_writer not implemented for transformed_structured_data_reader");
	}
}
