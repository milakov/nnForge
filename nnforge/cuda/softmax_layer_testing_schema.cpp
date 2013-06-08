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

#include "softmax_layer_testing_schema.h"

#include "../softmax_layer.h"
#include "softmax_layer_tester_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		softmax_layer_testing_schema::softmax_layer_testing_schema()
		{
		}

		softmax_layer_testing_schema::~softmax_layer_testing_schema()
		{
		}

		const boost::uuids::uuid& softmax_layer_testing_schema::get_uuid() const
		{
			return softmax_layer::layer_guid;
		}

		std::tr1::shared_ptr<layer_testing_schema> softmax_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema_smart_ptr(new softmax_layer_testing_schema());
		}

		layer_tester_cuda_smart_ptr softmax_layer_testing_schema::create_tester_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_tester_cuda_smart_ptr(new softmax_layer_tester_cuda());
		}
	}
}
