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

#include <mutex>
#include <vector>
#include <condition_variable>
#include <array>

namespace nnforge
{
	namespace cuda
	{
		class host_staged_cuda_communicator : public cuda_communicator
		{
		public:
			host_staged_cuda_communicator(int device_count);

			virtual ~host_staged_cuda_communicator() = default;

			virtual void enqueue_reduce_all(
				const char * name,
				int device_pos,
				cuda_linear_buffer_device::ptr data,
				cuda_stream::ptr stream);

		private:
			int device_count;
			int buffers_accumulated;

			std::mutex buffers_accumulated_mutex;
			std::vector<cuda_linear_buffer_host::ptr> src_data_list;
			std::array<cuda_linear_buffer_host::ptr, 2> reduced_data_list;
			std::string current_name;
			size_t current_size;

			int current_pack;

			bool run_distribute;
			std::mutex run_distribute_mutex;
			std::condition_variable run_distribute_condition;

			bool all_distribution_enqueued;
			std::mutex all_distribution_enqueued_mutex;
			std::condition_variable all_distribution_enqueued_condition;

			int threads_still_running;
			bool new_pack_can_run;
			std::mutex new_pack_can_run_mutex;
			std::condition_variable new_pack_can_run_condition;
		};
	}
}
