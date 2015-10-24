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

#include "plain_buffer.h"

#include <cstdlib>

namespace nnforge
{
	namespace plain
	{
		plain_buffer::plain_buffer(size_t size)
			: buf(0)
			, size(0)
		{
			buf = malloc(size);
			this->size = size;
		}

		plain_buffer::~plain_buffer()
		{
			free(buf);
		}

		void * plain_buffer::get_buf()
		{
			return buf;
		}

		const void * plain_buffer::get_buf() const
		{
			return buf;
		}

		size_t plain_buffer::get_size() const
		{
			return size;
		}

		plain_buffer::operator void *()
		{
			return get_buf();
		}

		plain_buffer::operator const void *() const
		{
			return get_buf();
		}

		plain_buffer::operator float *()
		{
			return (float *)(get_buf());
		}

		plain_buffer::operator const float *() const
		{
			return (float *)(get_buf());
		}

		plain_buffer::operator double *()
		{
			return (double *)(get_buf());
		}

		plain_buffer::operator const double *() const
		{
			return (double *)(get_buf());
		}

		plain_buffer::operator unsigned char *()
		{
			return (unsigned char *)(get_buf());
		}

		plain_buffer::operator const unsigned char *() const
		{
			return (unsigned char *)(get_buf());
		}

		plain_buffer::operator unsigned int *()
		{
			return (unsigned int *)(get_buf());
		}

		plain_buffer::operator const unsigned int *() const
		{
			return (unsigned int *)(get_buf());
		}

		plain_buffer::operator int *()
		{
			return (int *)(get_buf());
		}

		plain_buffer::operator const int *() const
		{
			return (int *)(get_buf());
		}
	}
}
