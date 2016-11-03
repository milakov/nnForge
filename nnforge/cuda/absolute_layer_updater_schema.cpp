/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "absolute_layer_updater_schema.h"

#include "../absolute_layer.h"
#include "absolute_layer_updater_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		layer_updater_schema::ptr absolute_layer_updater_schema::create_specific() const
		{
			return layer_updater_schema::ptr(new absolute_layer_updater_schema());
		}

		std::string absolute_layer_updater_schema::get_type_name() const
		{
			return absolute_layer::layer_type_name;
		}

		layer_updater_cuda::ptr absolute_layer_updater_schema::create_updater_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_updater_cuda::ptr(new absolute_layer_updater_cuda());
		}
	}
}
