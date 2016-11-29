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

#include "layer_testing_schema.h"

#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		layer_testing_schema::ptr layer_testing_schema::create(layer::const_ptr layer_schema) const
		{
			layer_testing_schema::ptr res = create_specific();

			res->layer_schema = layer_schema;

			return res;
		}

		layer_tester_cuda::ptr layer_testing_schema::create_tester(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			cuda_running_configuration::const_ptr cuda_config) const
		{
			layer_tester_cuda::ptr res = create_tester_specific(
				input_configuration_specific_list,
				output_configuration_specific,
				*cuda_config);

			res->configure(
				input_configuration_specific_list,
				output_configuration_specific,
				layer_schema,
				cuda_config);

			return res;
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_testing_schema::get_schema_buffers() const
		{
			return std::vector<cuda_linear_buffer_device::const_ptr>();
		}
	}
}
