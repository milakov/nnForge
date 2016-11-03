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
#include "cuda_linear_buffer_device.h"
#include "cuda_running_configuration.h"

#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class cuda_texture : public cuda_memobject
		{
		public:
			typedef std::shared_ptr<cuda_texture> ptr;
			typedef std::shared_ptr<const cuda_texture> const_ptr;

			cuda_texture(
				cuda_linear_buffer_device::const_ptr dev_smart_ptr,
				int vector_size = 1);

			virtual ~cuda_texture();

			operator cudaTextureObject_t () const;

			virtual size_t get_size() const;

		protected:
			cudaTextureObject_t tex;
			cuda_linear_buffer_device::const_ptr dev_smart_ptr;
		};
	}
}
