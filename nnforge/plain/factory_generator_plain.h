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

#pragma once

#include "../factory_generator.h"
#include "plain_running_configuration.h"

namespace nnforge
{
	namespace plain
	{
		class factory_generator_plain : public factory_generator
		{
		public:
			factory_generator_plain(
				float plain_max_global_memory_usage,
				int plain_openmp_thread_count);

			factory_generator_plain();

			~factory_generator_plain();

			virtual void initialize();

			virtual forward_propagation_factory::ptr create_forward_propagation_factory() const;

			virtual backward_propagation_factory::ptr create_backward_propagation_factory() const;

			virtual void info() const;

			virtual std::vector<float_option> get_float_options();

			virtual std::vector<int_option> get_int_options();

		protected:
			float plain_max_global_memory_usage;
			int plain_openmp_thread_count;

			plain_running_configuration::const_ptr plain_config;
		};
	}
}
