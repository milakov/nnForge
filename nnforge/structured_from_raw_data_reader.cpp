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

#include "structured_from_raw_data_reader.h"

namespace nnforge
{
	structured_from_raw_data_reader::structured_from_raw_data_reader(
		raw_data_reader::ptr raw_reader,
		raw_to_structured_data_transformer::ptr transformer)
		: raw_reader(raw_reader)
		, transformer(transformer)
		, transformer_sample_count(transformer->get_sample_count())
	{
	}

	structured_from_raw_data_reader::structured_from_raw_data_reader()
	{
	}

	structured_from_raw_data_reader::~structured_from_raw_data_reader()
	{
	}

	bool structured_from_raw_data_reader::read(
		unsigned int entry_id,
		float * data)
	{
		unsigned int original_entry_id = entry_id / transformer_sample_count;

		std::vector<unsigned char> raw_data;
		if (!raw_reader->raw_read(original_entry_id, raw_data))
			return false;

		unsigned int sample_id = entry_id - original_entry_id * transformer_sample_count;
		transformer->transform(sample_id, raw_data, data);
		return true;
	}

	bool structured_from_raw_data_reader::raw_read(
		unsigned int entry_id,
		std::vector<unsigned char>& all_elems)
	{
		return raw_reader->raw_read(entry_id, all_elems);
	}

	layer_configuration_specific structured_from_raw_data_reader::get_configuration() const
	{
		return transformer->get_configuration();
	}

	int structured_from_raw_data_reader::get_entry_count() const
	{
		return raw_reader->get_entry_count() * transformer_sample_count;
	}

	raw_data_writer::ptr structured_from_raw_data_reader::get_writer(nnforge_shared_ptr<std::ostream> out) const
	{
		return raw_reader->get_writer(out);
	}
}
