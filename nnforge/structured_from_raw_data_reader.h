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

#include "structured_data_reader.h"
#include "raw_data_reader.h"
#include "raw_to_structured_data_transformer.h"

namespace nnforge
{
	class structured_from_raw_data_reader : public structured_data_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_from_raw_data_reader> ptr;

		structured_from_raw_data_reader(
			raw_data_reader::ptr raw_reader,
			raw_to_structured_data_transformer::ptr transformer);

		virtual ~structured_from_raw_data_reader();

		virtual bool read(
			unsigned int entry_id,
			float * data);

		virtual bool raw_read(
			unsigned int entry_id,
			std::vector<unsigned char>& all_elems);

		virtual layer_configuration_specific get_configuration() const;

		virtual int get_entry_count() const;

		virtual raw_data_writer::ptr get_writer(nnforge_shared_ptr<std::ostream> out) const;

	protected:
		raw_data_reader::ptr raw_reader;
		raw_to_structured_data_transformer::ptr transformer;

	protected:
		structured_from_raw_data_reader();

	private:
		structured_from_raw_data_reader(const structured_from_raw_data_reader&);
		structured_from_raw_data_reader& operator =(const structured_from_raw_data_reader&);
	};
}
