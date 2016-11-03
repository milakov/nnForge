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
#include <memory>

namespace nnforge
{
	namespace plain
	{
		class plain_buffer
		{
		public:
			typedef std::shared_ptr<plain_buffer> ptr;
			typedef std::shared_ptr<const plain_buffer> const_ptr;

			plain_buffer(size_t size);

			virtual ~plain_buffer();

			// Size in bytes
			virtual size_t get_size() const;

			operator void *();

			operator const void *() const;

			operator float *();

			operator const float *() const;

			operator double *();

			operator const double *() const;

			operator unsigned char *();

			operator const unsigned char *() const;

			operator unsigned int *();

			operator const unsigned int *() const;

			operator int *();

			operator const int *() const;

		protected:
			void * get_buf();
			const void * get_buf() const;

		private:
			void * buf;
			size_t size;
		};
	}
}
