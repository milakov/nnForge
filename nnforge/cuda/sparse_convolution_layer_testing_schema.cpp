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

#include "sparse_convolution_layer_testing_schema.h"

#include "../sparse_convolution_layer.h"
#include "../neural_network_exception.h"
#include "sparse_fully_connected_1x1_layer_tester_cuda.h"
#include "sparse_fully_connected_layer_tester_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		sparse_convolution_layer_testing_schema::sparse_convolution_layer_testing_schema()
		{
		}

		sparse_convolution_layer_testing_schema::~sparse_convolution_layer_testing_schema()
		{
		}

		const boost::uuids::uuid& sparse_convolution_layer_testing_schema::get_uuid() const
		{
			return sparse_convolution_layer::layer_guid;
		}

		layer_testing_schema_smart_ptr sparse_convolution_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema_smart_ptr(new sparse_convolution_layer_testing_schema());
		}

		layer_tester_cuda_smart_ptr sparse_convolution_layer_testing_schema::create_tester_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			layer_tester_cuda_smart_ptr res;

			if (output_configuration_specific.get_neuron_count() == output_configuration_specific.feature_map_count)
			{
				if (input_configuration_specific.dimension_sizes == output_configuration_specific.dimension_sizes)
				{
					res = layer_tester_cuda_smart_ptr(new sparse_fully_connected_1x1_layer_tester_cuda());
				}
				else
				{
					res = layer_tester_cuda_smart_ptr(new sparse_fully_connected_layer_tester_cuda());
				}
			}
			else throw neural_network_exception("sparse convolutional layer has spatial fully connected tester implementation only in CUDA backend");

			return res;
		}
	}
}