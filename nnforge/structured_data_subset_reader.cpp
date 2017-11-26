/*
*  Copyright 2011-2017 Maxim Milakov
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

#include "structured_data_subset_reader.h"

#include <numeric>

#include "neural_network_exception.h"

namespace nnforge
{
	structured_data_subset_reader::structured_data_subset_reader(
		structured_data_reader::ptr original_reader,
		const std::vector<int>& samples_subset)
		: original_reader(original_reader)
		, entry_subset(entry_subset)
	{
		if (entry_subset.empty())
			throw neural_network_exception("Cannot create empty subset reader");
	}

	structured_data_subset_reader::structured_data_subset_reader(
		structured_data_reader::ptr original_reader,
		int fold_count,
		int fold_id,
		bool is_training)
		: original_reader(original_reader)
	{
		int original_entry_count = original_reader->get_entry_count();
		int min_entry_id = fold_id * original_entry_count / fold_count;
		int max_entry_id = (fold_id + 1) * original_entry_count / fold_count;
		int validation_entry_count = max_entry_id - min_entry_id;
		if (is_training)
		{
			entry_subset.resize(original_entry_count - validation_entry_count);
			std::iota(entry_subset.begin(), entry_subset.begin() + min_entry_id, 0);
			std::iota(entry_subset.begin() + min_entry_id, entry_subset.end(), max_entry_id);
		}
		else
		{
			entry_subset.resize(validation_entry_count);
			std::iota(entry_subset.begin(), entry_subset.end(), min_entry_id);
		}
		if(entry_subset.empty())
			throw neural_network_exception("Cannot create empty subset reader");
	}

	bool structured_data_subset_reader::read(
		unsigned int entry_id,
		float * data)
	{
		return original_reader->read(entry_subset[entry_id], data);
	}

	bool structured_data_subset_reader::raw_read(
		unsigned int entry_id,
		std::vector<unsigned char>& all_elems)
	{
		return original_reader->raw_read(entry_subset[entry_id], all_elems);
	}

	layer_configuration_specific structured_data_subset_reader::get_configuration() const
	{
		return original_reader->get_configuration();
	}

	int structured_data_subset_reader::get_entry_count() const
	{
		return static_cast<int>(entry_subset.size());
	}

	raw_data_writer::ptr structured_data_subset_reader::get_writer(std::shared_ptr<std::ostream> out) const
	{
		return original_reader->get_writer(out);
	}
}
