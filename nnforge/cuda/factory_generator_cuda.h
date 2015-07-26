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
#include "cuda_running_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class factory_generator_cuda : public factory_generator
		{
		public:
			factory_generator_cuda(
				int cuda_device_id,
				float cuda_max_global_memory_usage_ratio,
				unsigned int cuda_reserved_thread_count);

			factory_generator_cuda();

			~factory_generator_cuda();

			virtual void initialize();

			virtual forward_propagation_factory::ptr create_forward_propagation_factory() const;

			virtual void info() const;

			virtual std::vector<float_option> get_float_options();

			virtual std::vector<int_option> get_int_options();

		protected:
			int cuda_device_id;
			float cuda_max_global_memory_usage_ratio;
			int cuda_reserved_thread_count;

			cuda_running_configuration::const_ptr cuda_config;
		};
	}
}
