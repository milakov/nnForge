/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "structured_data_writer.h"

namespace nnforge
{
	void structured_data_writer::raw_write(
		const void * all_entry_data,
		size_t data_length)
	{
		write((const float *)all_entry_data);
	}

	void structured_data_writer::raw_write(
		unsigned int entry_id,
		const void * all_entry_data,
		size_t data_length)
	{
		write(entry_id, (const float *)all_entry_data);
	}
}
