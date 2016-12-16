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

#include "cuda_communicator.h"
#include "cuda_linear_buffer_host.h"

#include <vector>

#ifdef NNFORGE_USE_NCCL
#include <nccl.h>
#endif

namespace nnforge
{
	namespace cuda
	{
		class nccl_cuda_communicator : public cuda_communicator
		{
		public:
			nccl_cuda_communicator(int device_count);

			virtual ~nccl_cuda_communicator();

			virtual void enqueue_reduce_all(
				const char * name,
				int device_pos,
				cuda_linear_buffer_device::ptr data,
				cuda_stream::ptr stream);

		private:
#ifdef NNFORGE_USE_NCCL
			std::vector<bool> initialized_list;
			int device_count;
			ncclUniqueId communicator_id;
			std::vector<ncclComm_t> communicators;
#endif
		};
	}
}
