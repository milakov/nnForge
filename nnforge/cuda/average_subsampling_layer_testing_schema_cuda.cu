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

#include "average_subsampling_layer_testing_schema.h"

#include "../neural_network_exception.h"
#include "../average_subsampling_layer.h"
#include "average_subsampling_layer_tester_cuda.cuh"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		std::string average_subsampling_layer_testing_schema::get_type_name() const
		{
			return average_subsampling_layer::layer_type_name;
		}

		layer_testing_schema::ptr average_subsampling_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema::ptr(new average_subsampling_layer_testing_schema());
		}

		layer_tester_cuda::ptr average_subsampling_layer_testing_schema::create_tester_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			layer_tester_cuda::ptr res;

			switch (output_configuration_specific.dimension_sizes.size())
			{
				case 0:
					res = layer_tester_cuda::ptr(new average_subsampling_layer_tester_cuda<1>());
					break;
				case 1:
					res = layer_tester_cuda::ptr(new average_subsampling_layer_tester_cuda<1>());
					break;
				case 2:
					res = layer_tester_cuda::ptr(new average_subsampling_layer_tester_cuda<2>());
					break;
				case 3:
					res = layer_tester_cuda::ptr(new average_subsampling_layer_tester_cuda<3>());
					break;
				case 4:
					res = layer_tester_cuda::ptr(new average_subsampling_layer_tester_cuda<4>());
					break;
				default:
					throw neural_network_exception((boost::format("No CUDA tester for the average subsampling of %1% dimensions") % output_configuration_specific.dimension_sizes.size()).str());
			}

			return res;
		}
	}
}
