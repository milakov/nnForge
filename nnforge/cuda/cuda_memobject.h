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

#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class cuda_memobject
		{
		public:
			virtual ~cuda_memobject() = default;

			virtual size_t get_size() const = 0;

		protected:
			cuda_memobject() = default;

		private:
			cuda_memobject(const cuda_memobject&) = delete;
			cuda_memobject& operator =(const cuda_memobject&) = delete;
			
		};

		typedef std::shared_ptr<cuda_memobject> cuda_memobject_smart_ptr;
		typedef std::shared_ptr<const cuda_memobject> const_cuda_memobject_smart_ptr;
	}
}
