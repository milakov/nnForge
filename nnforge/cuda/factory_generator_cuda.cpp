/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "network_tester_cuda_factory.h"
#include "network_updater_cuda_factory.h"
#include "network_analyzer_cuda_factory.h"

#include <iostream>

namespace nnforge
{
	namespace cuda
	{
		factory_generator_cuda::factory_generator_cuda()
		{
		}

		factory_generator_cuda::~factory_generator_cuda()
		{
		}

		void factory_generator_cuda::initialize()
		{
			cuda_config = cuda_running_configuration_const_smart_ptr(new cuda_running_configuration(cuda_device_id, cuda_max_global_memory_usage_ratio));
		}

		network_tester_factory_smart_ptr factory_generator_cuda::create_tester_factory() const
		{
			return network_tester_factory_smart_ptr(new network_tester_cuda_factory(cuda_config));
		}

		network_updater_factory_smart_ptr factory_generator_cuda::create_updater_factory() const
		{
			return network_updater_factory_smart_ptr(new network_updater_cuda_factory(cuda_config));
		}

		network_analyzer_factory_smart_ptr factory_generator_cuda::create_analyzer_factory() const
		{
			return network_analyzer_factory_smart_ptr(new network_analyzer_cuda_factory(cuda_config));
		}

		std::vector<float_option> factory_generator_cuda::get_float_options()
		{
			std::vector<float_option> res;

			res.push_back(float_option("cuda_max_global_memory_usage_ratio,G", &cuda_max_global_memory_usage_ratio, 0.9F, "part of the global memory to be used by a single CUDA configuration."));

			return res;
		}

		std::vector<int_option> factory_generator_cuda::get_int_options()
		{
			std::vector<int_option> res;

			res.push_back(int_option("cuda_device_id,D", &cuda_device_id, 0, "CUDA device ID."));

			return res;
		}

		void factory_generator_cuda::info() const
		{
			std::cout << *cuda_config;
		}
	}
}
