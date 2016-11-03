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

#include "factory_generator_plain.h"

#include "forward_propagation_plain_factory.h"
#include "backward_propagation_plain_factory.h"

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nnforge
{
	namespace plain
	{
		factory_generator_plain::factory_generator_plain(
			float plain_max_global_memory_usage,
			int plain_openmp_thread_count)
			: plain_max_global_memory_usage(plain_max_global_memory_usage)
			, plain_openmp_thread_count(plain_openmp_thread_count)
		{
		}

		void factory_generator_plain::initialize()
		{
			plain_config = plain_running_configuration::const_ptr(new plain_running_configuration(
				plain_openmp_thread_count,
				plain_max_global_memory_usage));
		}

		forward_propagation_factory::ptr factory_generator_plain::create_forward_propagation_factory() const
		{
			return forward_propagation_factory::ptr(new forward_propagation_plain_factory(plain_config));
		}

		backward_propagation_factory::ptr factory_generator_plain::create_backward_propagation_factory() const
		{
			return backward_propagation_factory::ptr(new backward_propagation_plain_factory(plain_config));
		}

		std::vector<float_option> factory_generator_plain::get_float_options()
		{
			std::vector<float_option> res;

			res.push_back(float_option("plain_max_global_memory_usage,M", &plain_max_global_memory_usage, 0.5F, "memory to be used by single plain configuration, in GB."));

			return res;
		}

		std::vector<int_option> factory_generator_plain::get_int_options()
		{
			std::vector<int_option> res;

			#ifdef _OPENMP
			res.push_back(int_option("plain_openmp_thread_count", &plain_openmp_thread_count, omp_get_max_threads(), "count of threads to be used in OpenMP."));
			#endif

			return res;
		}

		void factory_generator_plain::info() const
		{
			std::cout << *plain_config;
		}
	}
}
