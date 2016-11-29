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

#include <cuda_runtime.h>
#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class cuda_stream
		{
		public:
			typedef std::shared_ptr<cuda_stream> ptr;

			cuda_stream();

			cuda_stream(int priority);

			virtual ~cuda_stream();

			operator cudaStream_t ();

		protected:
			cudaStream_t stream;

		private:
			cuda_stream(const cuda_stream&) = delete;
			cuda_stream& operator =(const cuda_stream&) = delete;
		};
	}
}
