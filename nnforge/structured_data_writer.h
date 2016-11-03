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

#pragma once

#include "raw_data_writer.h"

#include <memory>

namespace nnforge
{
	class structured_data_writer : public raw_data_writer
	{
	public:
		typedef std::shared_ptr<structured_data_writer> ptr;

		virtual ~structured_data_writer() = default;

		virtual void raw_write(
			const void * all_entry_data,
			size_t data_length);

		virtual void raw_write(
			unsigned int entry_id,
			const void * all_entry_data,
			size_t data_length);

		virtual void write(const float * neurons) = 0;

		virtual void write(
			unsigned int entry_id,
			const float * neurons) = 0;

	protected:
		structured_data_writer() = default;

	private:
		structured_data_writer(const structured_data_writer&) = delete;
		structured_data_writer& operator =(const structured_data_writer&) = delete;
	};
}
