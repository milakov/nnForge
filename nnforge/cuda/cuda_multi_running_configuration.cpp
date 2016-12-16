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

#include "cuda_multi_running_configuration.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/format.hpp>
#include <thread>
#include <iostream>

#include "neural_network_cuda_exception.h"
#include "host_staged_cuda_communicator.h"
#include "nccl_cuda_communicator.h"

namespace nnforge
{
	namespace cuda
	{
		cuda_multi_running_configuration::cuda_multi_running_configuration(
			const std::vector<unsigned int>& device_id_list,
			float max_global_memory_usage_ratio,
			unsigned int reserved_thread_count,
			bool dont_share_buffers,
			bool single_command_stream,
			unsigned int optimize_action_graph_assumed_chunk_size,
			float cuda_fixed_working_buffers_ratio,
			const std::string& communicator_type)
			: reserved_thread_count(reserved_thread_count)
			, dont_share_buffers(dont_share_buffers)
			, single_command_stream(single_command_stream)
			, optimize_action_graph_assumed_chunk_size(optimize_action_graph_assumed_chunk_size)
		{
			update_parameters();

			cuda_communicator::ptr communicator;
			if (communicator_type == "host_staged")
				communicator = cuda_communicator::ptr(new host_staged_cuda_communicator(static_cast<int>(device_id_list.size())));
			else if (communicator_type == "nccl")
				communicator = cuda_communicator::ptr(new nccl_cuda_communicator(static_cast<int>(device_id_list.size())));
			else
				throw neural_network_exception((boost::format("Unknown communicator type specified: %1%") % communicator_type).str());

			for(int device_pos = 0; device_pos < device_id_list.size(); ++device_pos)
			{
				unsigned int device_id = device_id_list[device_pos];
				if (device_id >= static_cast<unsigned int>(cuda_device_count))
					throw neural_network_exception((boost::format("Device ID %1% specified while %2% devices are available") % device_id % cuda_device_count).str());

				cuda_config_list.push_back(cuda_running_configuration::ptr(new cuda_running_configuration(
					device_id,
					max_global_memory_usage_ratio,
					cuda_fixed_working_buffers_ratio,
					device_pos,
					communicator)));
			}
		}

		threadpool_job_runner::ptr cuda_multi_running_configuration::get_job_runner() const
		{
			return job_runner;
		}

		bool cuda_multi_running_configuration::is_dont_share_buffers() const
		{
			return dont_share_buffers;
		}

		bool cuda_multi_running_configuration::is_single_command_stream() const
		{
			return single_command_stream;
		}

		void cuda_multi_running_configuration::update_parameters()
		{
			cuda_safe_call(cudaDriverGetVersion(&driver_version));
			cuda_safe_call(cudaRuntimeGetVersion(&runtime_version));

			cuda_safe_call(cudaGetDeviceCount(&cuda_device_count));
			if (cuda_device_count <= 0)
				throw neural_network_exception("No CUDA capable devices are found");

			unsigned int core_count = std::thread::hardware_concurrency();
			job_runner = threadpool_job_runner::ptr(new threadpool_job_runner(core_count > reserved_thread_count ? core_count - reserved_thread_count : 1));
		}

		std::vector<unsigned int> cuda_multi_running_configuration::get_default_device_id_list()
		{
			try
			{
				int cuda_device_count;
				cudaGetDeviceCount(&cuda_device_count);
				if (cuda_device_count == 0)
					return std::vector<unsigned int>(1, 0);
				else
				{
					std::vector<unsigned int> res;

					cudaDeviceProp device_prop;
					cuda_safe_call(cudaGetDeviceProperties(&device_prop, 0));
					int compute_capability_major = device_prop.major;
					int compute_capability_minor = device_prop.minor;
					int multiprocessor_count = device_prop.multiProcessorCount;
					res.push_back(0);
					for(int i = 1; i < cuda_device_count; ++i)
					{
						cuda_safe_call(cudaGetDeviceProperties(&device_prop, i));
						if ((compute_capability_major != device_prop.major)
							|| (compute_capability_minor != device_prop.minor)
							|| (multiprocessor_count != device_prop.multiProcessorCount))
							break;
						res.push_back(i);
					}

					return res;
				}
			}
			catch (neural_network_cuda_exception&)
			{
				return std::vector<unsigned int>(1, 0);
			}
		}

		std::ostream& operator<< (std::ostream& out, const cuda_multi_running_configuration& multi_running_configuration)
		{
			out << "--- CUDA versions ---" << std::endl;
			out << "Driver version = " << multi_running_configuration.driver_version / 1000 << "." << (multi_running_configuration.driver_version % 100) / 10 << std::endl;
			out << "Runtime version = " << multi_running_configuration.runtime_version / 1000 << "." << (multi_running_configuration.runtime_version % 100) / 10 << std::endl;

			out << "--- Settings ---" << std::endl;
			out << "Don't share buffers = " << multi_running_configuration.dont_share_buffers << std::endl;
			out << "Use single command stream = " << multi_running_configuration.single_command_stream << std::endl;
			out << "Assumed chunk size when optimizing action graph = " << multi_running_configuration.optimize_action_graph_assumed_chunk_size << std::endl;
			out << "Threads reserved for CUDA sync (others will be used for on-the-fly data processing by job runner) = " << multi_running_configuration.reserved_thread_count << std::endl;

			out << "--- Status ---" << std::endl;
			out << "Job runner thread count = " << multi_running_configuration.get_job_runner()->thread_count << std::endl;

			for(auto it: multi_running_configuration.cuda_config_list)
				std::cout << *it;

			return out;
		}
	}
}
