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

#include <memory>

#include "cuda_stream.h"
#include "cuda_linear_buffer_device.h"

namespace nnforge
{
	namespace cuda
	{
		class cuda_communicator
		{
		public:
			typedef std::shared_ptr<cuda_communicator> ptr;
			typedef std::shared_ptr<const cuda_communicator> const_ptr;

			virtual ~cuda_communicator() = default;

			virtual void enqueue_reduce_all(
				const char * name,
				int device_pos,
				cuda_linear_buffer_device::ptr data,
				cuda_stream::ptr stream) = 0;

		protected:
			cuda_communicator() = default;

		private:
			cuda_communicator(const cuda_communicator&) = delete;
			cuda_communicator& operator =(const cuda_communicator&) = delete;
		};
	}
}
