/*
 *  Copyright 2011-2013 Maxim Milakov
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
			factory_generator_cuda();

			~factory_generator_cuda();

			virtual void initialize();

			virtual network_tester_factory_smart_ptr create_tester_factory() const;

			virtual network_updater_factory_smart_ptr create_updater_factory() const;

			virtual hessian_calculator_factory_smart_ptr create_hessian_factory() const;

			virtual void info() const;

			virtual std::vector<float_option> get_float_options();

			virtual std::vector<int_option> get_int_options();

		protected:
			int cuda_device_id;
			float cuda_max_global_memory_usage_ratio;

			cuda_running_configuration_const_smart_ptr cuda_config;
		};
	}
}
