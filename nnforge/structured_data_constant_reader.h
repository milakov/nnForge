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
#include "nn_types.h"

namespace nnforge
{
	class structured_data_constant_reader : public structured_data_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_data_constant_reader> ptr;

		// The constructor modifies input_stream to throw exceptions in case of failure
		structured_data_constant_reader(
			float val,
			const layer_configuration_specific& config,
			int entry_count = -1);

		virtual ~structured_data_constant_reader();

		virtual bool read(
			unsigned int entry_id,
			float * data);

		virtual layer_configuration_specific get_configuration() const;

		virtual int get_entry_count() const;

		virtual raw_data_writer::ptr get_writer(nnforge_shared_ptr<std::ostream> out) const;

	protected:
		float val;
		layer_configuration_specific config;
		int entry_count;

	private:
		structured_data_constant_reader(const structured_data_constant_reader&);
		structured_data_constant_reader& operator =(const structured_data_constant_reader&);
	};
}
