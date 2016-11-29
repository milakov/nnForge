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

#include "host_staged_cuda_communicator.h"

#include "neural_network_cuda_exception.h"
#include "cuda_profiling.h"

#include <boost/format.hpp>
#include <algorithm>
#include <cstring>

namespace nnforge
{
	namespace cuda
	{
		host_staged_cuda_communicator::host_staged_cuda_communicator(int device_count)
			: device_count(device_count)
			, buffers_accumulated(0)
			, new_pack_can_run(true)
			, src_data_list(device_count)
			, current_pack(0)
		{
		}

		void host_staged_cuda_communicator::enqueue_reduce_all(
			const char * name,
			int device_pos,
			cuda_linear_buffer_device::ptr data,
			cuda_stream::ptr stream)
		{
			std::string profiling_str = (boost::format("enqueue_reduce_all for %1%") % name).str();
			PUSH_RANGE(profiling_str.c_str(), 8);

			// Wait for all threads from previous enqueue to leave function
			{
				std::unique_lock<std::mutex> lock(new_pack_can_run_mutex);
				while (!new_pack_can_run)
					new_pack_can_run_condition.wait(lock);
			}

			size_t elem_count = data->get_size() / sizeof(float);

			if ((!src_data_list[device_pos]) || (src_data_list[device_pos]->get_size() < data->get_size()))
				src_data_list[device_pos] = cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(data->get_size()));
			cuda_safe_call(cudaMemcpyAsync(*src_data_list[device_pos], *data, data->get_size(), cudaMemcpyDeviceToHost, *stream));
			cuda_safe_call(cudaStreamSynchronize(*stream));

			{
				std::lock_guard<std::mutex> lock(buffers_accumulated_mutex);

				if (buffers_accumulated == 0)
				{
					current_pack = 1 - current_pack;

					if ((!reduced_data_list[current_pack]) || (reduced_data_list[current_pack]->get_size() < data->get_size()))
						reduced_data_list[current_pack] = cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(data->get_size()));
					current_name = name;
					current_size = data->get_size();

					float * dst = *reduced_data_list[current_pack];
					float * src = *src_data_list[device_pos];
					memcpy(dst, src, current_size);

					run_distribute = false;
					all_distribution_enqueued = false;
				}
				else
				{
					if (current_name != name)
						throw neural_network_exception((boost::format("reduce_all is requested for %1% while another one for %2% is in progress") % name % current_name).str());
					if (current_size != data->get_size())
						throw neural_network_exception((boost::format("reduce_all is requested for %1% with data of size %2% bytes while it was previously initialized with %3% bytes") % name % data->get_size() % current_size).str());

					float * dst = *reduced_data_list[current_pack];
					float * src = *src_data_list[device_pos];
					std::transform(dst, dst + elem_count, src, dst, std::plus<float>());
				}

				++buffers_accumulated;

				if (buffers_accumulated == device_count)
				{
					{
						std::lock_guard<std::mutex> lock(run_distribute_mutex);
						new_pack_can_run = false;
						threads_still_running = device_count;
						run_distribute = true;
					}
					run_distribute_condition.notify_all();
				}
			}

			// Wait for all threads to read and reduce data
			{
				std::unique_lock<std::mutex> lock(run_distribute_mutex);
				while (!run_distribute)
					run_distribute_condition.wait(lock);
			}

			cuda_safe_call(cudaMemcpyAsync(*data, *reduced_data_list[current_pack], data->get_size(), cudaMemcpyDeviceToHost, *stream));
			// cuda_safe_call(cudaStreamSynchronize(*stream));

			{
				std::lock_guard<std::mutex> lock(buffers_accumulated_mutex);
				--buffers_accumulated;

				if (buffers_accumulated == 0)
				{
					{
						std::lock_guard<std::mutex> lock(all_distribution_enqueued_mutex);
						all_distribution_enqueued = true;
					}
					all_distribution_enqueued_condition.notify_all();
				}
			}

			// Wait for all threads to finish H2D copy
			{
				std::unique_lock<std::mutex> lock(all_distribution_enqueued_mutex);
				while (!all_distribution_enqueued)
					all_distribution_enqueued_condition.wait(lock);
			}

			{
				std::lock_guard<std::mutex> lock(new_pack_can_run_mutex);

				--threads_still_running;
				{
					if (threads_still_running == 0)
					{
						new_pack_can_run = true;
					}
					new_pack_can_run_condition.notify_all();
				}
			}

			POP_RANGE;
		}
	}
}
