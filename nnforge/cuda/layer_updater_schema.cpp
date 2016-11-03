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

#include "layer_updater_schema.h"

namespace nnforge
{
	namespace cuda
	{
		std::shared_ptr<layer_updater_schema> layer_updater_schema::create(
			layer::const_ptr layer_schema,
			cuda_running_configuration::const_ptr cuda_config) const
		{
			std::shared_ptr<layer_updater_schema> res = create_specific();

			res->layer_schema = layer_schema;
			res->cuda_config = cuda_config;

			return res;
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_updater_schema::get_schema_buffers() const
		{
			return std::vector<cuda_linear_buffer_device::const_ptr>();
		}

		layer_updater_cuda::ptr layer_updater_schema::create_updater(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			const std::set<layer_action>& actions) const
		{
			layer_updater_cuda::ptr res = create_updater_specific(
				input_configuration_specific_list,
				output_configuration_specific);

			res->configure(
				input_configuration_specific_list,
				output_configuration_specific,
				layer_schema,
				cuda_config,
				actions);

			return res;
		}
	}
}
