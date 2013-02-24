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

#include "buffer_plain_size_configuration.h"

namespace nnforge
{
	namespace plain
	{
		buffer_plain_size_configuration::buffer_plain_size_configuration()
			: constant_buffer_size(0)
			, per_entry_buffer_size(0)
		{
		}

		void buffer_plain_size_configuration::add_constant_buffer(size_t buffer_size)
		{
			constant_buffer_size += buffer_size;
		}

		void buffer_plain_size_configuration::add_per_entry_buffer(size_t buffer_size)
		{
			per_entry_buffer_size += buffer_size;
		}
	}
}
