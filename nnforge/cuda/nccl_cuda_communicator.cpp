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

#include "nccl_cuda_communicator.h"

#include "neural_network_cuda_exception.h"
#include "cuda_profiling.h"

#include <boost/format.hpp>
#include <algorithm>
#include <cstring>

#include "../neural_network_exception.h"

namespace nnforge
{
	namespace cuda
	{
#ifdef NNFORGE_USE_NCCL
		class nccl_exception : public neural_network_exception
		{
		public:
			nccl_exception(
				ncclResult_t error_code,
				const char * filename,
				int line_number)
			: neural_network_exception((boost::format("NCCL error: %1%") % ncclGetErrorString(error_code)).str(), filename, line_number)
			{
			}
		};
		#define nccl_safe_call(callstr) {ncclResult_t error_code = callstr; if (error_code != ncclSuccess) throw nnforge::cuda::nccl_exception(error_code, __FILE__, __LINE__);}
#endif

		nccl_cuda_communicator::nccl_cuda_communicator(int device_count)
#ifdef NNFORGE_USE_NCCL
		: initialized_list(device_count, false)
		, device_count(device_count)
#endif
		{
#ifdef NNFORGE_USE_NCCL
			nccl_safe_call(ncclGetUniqueId(&communicator_id));
			communicators.resize(device_count);
#else
			throw neural_network_exception("Framework built without NCCL support");
#endif
		}

		nccl_cuda_communicator::~nccl_cuda_communicator()
		{
#ifdef NNFORGE_USE_NCCL
			for(int i = 0; i < device_count; ++i)
			{
				if (initialized_list[i])
					ncclCommDestroy(communicators[i]);
			}
#endif
		}

		void nccl_cuda_communicator::enqueue_reduce_all(
			const char * name,
			int device_pos,
			cuda_linear_buffer_device::ptr data,
			cuda_stream::ptr stream)
		{
#ifdef NNFORGE_USE_NCCL
			if (!initialized_list[device_pos])
			{
				PUSH_RANGE("NCCL communicator init", 9);
				nccl_safe_call(ncclCommInitRank(&communicators[device_pos], device_count, communicator_id, device_pos));
				initialized_list[device_pos] = true;
				POP_RANGE;
			}

			std::string profiling_str = (boost::format("enqueue_reduce_all for %1%") % name).str();
			PUSH_RANGE(profiling_str.c_str(), 8);

			nccl_safe_call(ncclAllReduce((float *)(*data), (float *)(*data), data->get_size() / sizeof(float), ncclFloat, ncclSum, communicators[device_pos], *stream));

			POP_RANGE;
#endif
		}
	}
}
