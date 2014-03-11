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

#include "factory_generator_plain.h"

#include "network_tester_plain_factory.h"
#include "hessian_calculator_plain_factory.h"
#include "network_updater_plain_factory.h"
#include "network_analyzer_plain_factory.h"

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nnforge
{
	namespace plain
	{
		factory_generator_plain::factory_generator_plain()
			#ifdef _OPENMP
			: plain_openmp_thread_count(omp_get_max_threads())
			#else
			: plain_openmp_thread_count(1)
			#endif
			, plain_max_global_memory_usage(0.5F)
		{
		}

		factory_generator_plain::~factory_generator_plain()
		{
		}

		void factory_generator_plain::initialize()
		{
			plain_config = plain_running_configuration_const_smart_ptr(new plain_running_configuration(plain_openmp_thread_count, plain_max_global_memory_usage));
		}

		network_tester_factory_smart_ptr factory_generator_plain::create_tester_factory() const
		{
			return network_tester_factory_smart_ptr(new network_tester_plain_factory(plain_config));
		}

		network_updater_factory_smart_ptr factory_generator_plain::create_updater_factory() const
		{
			return network_updater_factory_smart_ptr(new network_updater_plain_factory(plain_config));
		}

		hessian_calculator_factory_smart_ptr factory_generator_plain::create_hessian_factory() const
		{
			return hessian_calculator_factory_smart_ptr(new hessian_calculator_plain_factory(plain_config));
		}

		network_analyzer_factory_smart_ptr factory_generator_plain::create_analyzer_factory() const
		{
			return network_analyzer_factory_smart_ptr(new network_analyzer_plain_factory(plain_config));
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
