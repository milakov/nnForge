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

#include "parametric_rectified_linear_layer_testing_schema.h"

#include "../parametric_rectified_linear_layer.h"
#include "parametric_rectified_linear_layer_tester_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		parametric_rectified_linear_layer_testing_schema::parametric_rectified_linear_layer_testing_schema()
		{
		}

		parametric_rectified_linear_layer_testing_schema::~parametric_rectified_linear_layer_testing_schema()
		{
		}

		const boost::uuids::uuid& parametric_rectified_linear_layer_testing_schema::get_uuid() const
		{
			return parametric_rectified_linear_layer::layer_guid;
		}

		layer_testing_schema_smart_ptr parametric_rectified_linear_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema_smart_ptr(new parametric_rectified_linear_layer_testing_schema());
		}

		layer_tester_cuda_smart_ptr parametric_rectified_linear_layer_testing_schema::create_tester_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_tester_cuda_smart_ptr(new parametric_rectified_linear_layer_tester_cuda());
		}
	}
}
