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

#include "../network_tester_factory.h"
#include "cuda_running_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class network_tester_cuda_factory : public network_tester_factory
		{
		public:
			network_tester_cuda_factory(cuda_running_configuration_const_smart_ptr opencl_config);

			virtual ~network_tester_cuda_factory();

			virtual network_tester_smart_ptr create(
				network_schema_smart_ptr schema,
				const_data_scale_params_smart_ptr scale_params) const;

		protected:
			cuda_running_configuration_const_smart_ptr cuda_config;
		};
	}
}
