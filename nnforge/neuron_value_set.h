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

#pragma once

#include <vector>
#include <ostream>
#include <istream>
#include <string>
#include <map>
#include <boost/uuid/uuid.hpp>

#include "nn_types.h"

namespace nnforge
{
	class neuron_value_set
	{
	public:
		enum merge_type_enum
		{
			merge_average,
			merge_median
		};

		typedef nnforge_shared_ptr<neuron_value_set> ptr;
		typedef nnforge_shared_ptr<const neuron_value_set> const_ptr;

		neuron_value_set(unsigned int neuron_count);

		neuron_value_set(
			unsigned int neuron_count,
			unsigned int entry_count);

		neuron_value_set(
			const std::vector<neuron_value_set::const_ptr>& source_neuron_value_set_list,
			merge_type_enum merge_type);

		neuron_value_set(
			const std::vector<std::pair<neuron_value_set::const_ptr, float> >& source_neuron_value_set_list);

		void add_entry(const float * new_data);

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		nnforge_shared_ptr<std::vector<float> > get_average() const;

		const boost::uuids::uuid& get_uuid() const;

	public:
		unsigned int neuron_count;
		std::vector<nnforge_shared_ptr<std::vector<float> > > neuron_value_list;

	private:
		static const boost::uuids::uuid neuron_value_set_guid;
	};
}
