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

#include "cuda_memobject.h"

#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class cuda_array : public cuda_memobject
		{
		public:
			cuda_array(
				unsigned int width,
				unsigned int height,
				unsigned int layer_count);

			cuda_array(
				unsigned int width,
				unsigned int layer_count);

			virtual ~cuda_array();

			operator cudaArray_const_t () const;

			operator cudaArray_t ();

			virtual size_t get_size() const;

		protected:
			cudaArray_t arr;
		};

		typedef nnforge_shared_ptr<cuda_array> cuda_array_smart_ptr;
	}
}
