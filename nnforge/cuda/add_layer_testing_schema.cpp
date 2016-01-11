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

#include "add_layer_testing_schema.h"

#include "../add_layer.h"
#include "add_layer_tester_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		add_layer_testing_schema::add_layer_testing_schema()
		{
		}

		add_layer_testing_schema::~add_layer_testing_schema()
		{
		}

		std::string add_layer_testing_schema::get_type_name() const
		{
			return add_layer::layer_type_name;
		}

		layer_testing_schema::ptr add_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema::ptr(new add_layer_testing_schema());
		}

		layer_tester_cuda::ptr add_layer_testing_schema::create_tester_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_tester_cuda::ptr(new add_layer_tester_cuda());
		}
	}
}
