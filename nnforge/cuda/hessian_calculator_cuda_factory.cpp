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

#include "hessian_calculator_cuda_factory.h"

#include "hessian_calculator_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		hessian_calculator_cuda_factory::hessian_calculator_cuda_factory(cuda_running_configuration_const_smart_ptr cuda_config)
			: cuda_config(cuda_config)
		{
		}

		hessian_calculator_cuda_factory::~hessian_calculator_cuda_factory()
		{
		}

		hessian_calculator_smart_ptr hessian_calculator_cuda_factory::create(network_schema_smart_ptr schema) const
		{
			return hessian_calculator_smart_ptr(new hessian_calculator_cuda(schema, cuda_config));
		}
	}
}
