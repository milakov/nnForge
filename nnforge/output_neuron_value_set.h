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

#pragma once

#include <vector>
#include <ostream>
#include <istream>
#include <boost/uuid/uuid.hpp>

#include "nn_types.h"

namespace nnforge
{
	class output_neuron_value_set
	{
	public:
		enum merge_type_enum
		{
			merge_average,
			merge_median
		};

		output_neuron_value_set();

		output_neuron_value_set(
			unsigned int entry_count,
			unsigned int neuron_count);

		output_neuron_value_set(
			const std::vector<nnforge_shared_ptr<output_neuron_value_set> >& source_output_neuron_value_set_list,
			merge_type_enum merge_type);

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		void clamp(
			float min_val,
			float max_val);

		void compact(unsigned int sample_count);

		const boost::uuids::uuid& get_uuid() const;

	public:
		std::vector<std::vector<float> > neuron_value_list;

	private:
		static const boost::uuids::uuid output_neuron_value_set_guid;
	};

	typedef nnforge_shared_ptr<output_neuron_value_set> output_neuron_value_set_smart_ptr;
}
