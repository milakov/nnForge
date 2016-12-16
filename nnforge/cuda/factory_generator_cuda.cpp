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

#include "factory_generator_cuda.h"

#include "forward_propagation_cuda_factory.h"
#include "backward_propagation_cuda_factory.h"

#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>

namespace nnforge
{
	namespace cuda
	{
		factory_generator_cuda::factory_generator_cuda(
			const std::string& cuda_device_id_list_str,
			float cuda_max_global_memory_usage_ratio,
			unsigned int cuda_reserved_thread_count,
			bool cuda_dont_share_buffers,
			bool cuda_single_command_stream,
			unsigned int cuda_optimize_action_graph_assumed_chunk_size,
			float cuda_fixed_working_buffers_ratio,
			const std::string& communicator_type)
			: cuda_device_id_list_str(cuda_device_id_list_str)
			, cuda_max_global_memory_usage_ratio(cuda_max_global_memory_usage_ratio)
			, cuda_reserved_thread_count(cuda_reserved_thread_count)
			, cuda_dont_share_buffers(cuda_dont_share_buffers)
			, cuda_single_command_stream(cuda_single_command_stream)
			, cuda_optimize_action_graph_assumed_chunk_size(cuda_optimize_action_graph_assumed_chunk_size)
			, cuda_fixed_working_buffers_ratio(cuda_fixed_working_buffers_ratio)
			, communicator_type(communicator_type)
		{
		}

		void factory_generator_cuda::initialize()
		{
			std::vector<unsigned int> cuda_device_id_list;
			{
				std::vector<std::string> strs;
				boost::split(strs, cuda_device_id_list_str, boost::is_any_of(","));
				for(const auto& elem: strs)
					cuda_device_id_list.push_back(atol(elem.c_str()));
			}

			cuda_multi_config = cuda_multi_running_configuration::const_ptr(new cuda_multi_running_configuration(
				cuda_device_id_list,
				cuda_max_global_memory_usage_ratio,
				cuda_reserved_thread_count,
				cuda_dont_share_buffers,
				cuda_single_command_stream,
				cuda_optimize_action_graph_assumed_chunk_size,
				cuda_fixed_working_buffers_ratio,
				communicator_type));
		}

		forward_propagation_factory::ptr factory_generator_cuda::create_forward_propagation_factory() const
		{
			return forward_propagation_factory::ptr(new forward_propagation_cuda_factory(cuda_multi_config));
		}

		backward_propagation_factory::ptr factory_generator_cuda::create_backward_propagation_factory() const
		{
			return backward_propagation_factory::ptr(new backward_propagation_cuda_factory(cuda_multi_config));
		}

		std::vector<float_option> factory_generator_cuda::get_float_options()
		{
			std::vector<float_option> res;

			res.push_back(float_option("cuda_max_global_memory_usage_ratio,G", &cuda_max_global_memory_usage_ratio, 0.9F, "Part of the global memory to be used by a single CUDA configuration. Set to smaller value if the device is used for graphics as well"));
			res.push_back(float_option("cuda_fixed_working_buffers_ratio", &cuda_fixed_working_buffers_ratio, 0.4F, "Part of memory used by app, which is allocated to working buffers (independent of batch size). This parameter affects the performance, you might also need to reduce it when running large models on GPU with small RAM"));

			return res;
		}

		std::vector<int_option> factory_generator_cuda::get_int_options()
		{
			std::vector<int_option> res;

			res.push_back(int_option("cuda_reserved_thread_count", &cuda_reserved_thread_count, 1, "The number of hw threads not used for input data processing"));
			res.push_back(int_option("cuda_optimize_action_graph_assumed_chunk_size", &cuda_optimize_action_graph_assumed_chunk_size, 32, "Assumed chunk size when optimizing action graph"));

			return res;
		}

		std::vector<bool_option> factory_generator_cuda::get_bool_options()
		{
			std::vector<bool_option> res;

			res.push_back(bool_option("cuda_dont_share_buffers", &cuda_dont_share_buffers, false, "Don't share buffers between layer. Switch it on if you suspect a bug in there"));
			res.push_back(bool_option("cuda_single_command_stream", &cuda_single_command_stream, false, "Use single stream for kernels. Switch it on if you suspect a bug when kernels are submitted into multiple streams"));

			return res;
		}

		std::vector<string_option> factory_generator_cuda::get_string_options()
		{
			std::vector<string_option> res;

			std::stringstream default_device_id_list_str;
			auto default_device_id_list = cuda_multi_running_configuration::get_default_device_id_list();
			for(size_t i = 0; i < default_device_id_list.size(); ++i)
			{
				if(i != 0)
					default_device_id_list_str << ",";
				default_device_id_list_str << default_device_id_list[i];
			}
			res.push_back(string_option("cuda_device_id,D", &cuda_device_id_list_str, default_device_id_list_str.str().c_str(), "Comma-separated list of CUDA device IDs"));

#ifdef NNFORGE_USE_NCCL
			res.push_back(string_option("cuda_communicator_type", &communicator_type, "host_staged", "Type of the communicator for multi-gpu transfers (host_staged, nccl)"));
#else
			res.push_back(string_option("cuda_communicator_type", &communicator_type, "host_staged", "Type of the communicator for multi-gpu transfers (host_staged)"));
#endif

			return res;
		}

		void factory_generator_cuda::info() const
		{
			std::cout << *cuda_multi_config;
		}
	}
}
