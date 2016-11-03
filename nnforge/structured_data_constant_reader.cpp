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

#include "structured_data_constant_reader.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	structured_data_constant_reader::structured_data_constant_reader(
		float val,
		const layer_configuration_specific& config,
		int entry_count)
		: val(val)
		, config(config)
		, entry_count(entry_count)
	{
	}

	bool structured_data_constant_reader::read(
		unsigned int entry_id,
		float * data)
	{
		if ((entry_count >= 0) && (entry_id >= static_cast<unsigned int>(entry_count)))
			return false;

		std::fill_n(data, config.get_neuron_count(), val);

		return true;
	}

	layer_configuration_specific structured_data_constant_reader::get_configuration() const
	{
		return config;
	}

	int structured_data_constant_reader::get_entry_count() const
	{
		return entry_count;
	}

	raw_data_writer::ptr structured_data_constant_reader::get_writer(std::shared_ptr<std::ostream> out) const
	{
		throw std::runtime_error("get_writer not implemented for structured_data_constant_reader");
	}
}
