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

#include "cuda_memobject.h"

#include <memory>
#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class cuda_linear_buffer : public cuda_memobject
		{
		public:
			virtual ~cuda_linear_buffer() = default;

			operator void *();

			operator const void *() const;

			operator float *();

			operator const float *() const;

			operator double *();

			operator const double *() const;

			operator float2 *();

			operator const float2 *() const;

			operator float4 *();

			operator const float4 *() const;

			operator unsigned char *();

			operator const unsigned char *() const;

			operator uchar4 *();

			operator const uchar4 *() const;

			operator unsigned int *();

			operator const unsigned int *() const;

			operator int *();

			operator const int *() const;

			operator uint4 *();

			operator const uint4 *() const;

			operator int4 *();

			operator const int4 *() const;

		protected:
			cuda_linear_buffer() = default;

			virtual void * get_buf() = 0;
			virtual const void * get_buf() const = 0;
		};

		typedef std::shared_ptr<cuda_linear_buffer> cuda_linear_buffer_smart_ptr;
		typedef std::shared_ptr<const cuda_linear_buffer> const_cuda_linear_buffer_smart_ptr;
	}
}
