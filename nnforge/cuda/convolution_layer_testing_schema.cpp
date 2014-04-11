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

#include "convolution_layer_testing_schema.h"

#include "../convolution_layer.h"
#include "../neural_network_exception.h"
#include "convolution_1d_layer_tester_cuda_fermi.h"
#include "convolution_1d_layer_tester_cuda_kepler.h"
#include "convolution_2d_layer_tester_cuda_fermi.h"
#include "convolution_2d_layer_tester_cuda_kepler.h"
#include "convolution_3d_layer_tester_cuda_fermi.h"
#include "convolution_3d_layer_tester_cuda_kepler.h"
#include "fully_connected_layer_tester_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		convolution_layer_testing_schema::convolution_layer_testing_schema()
		{
		}

		convolution_layer_testing_schema::~convolution_layer_testing_schema()
		{
		}

		const boost::uuids::uuid& convolution_layer_testing_schema::get_uuid() const
		{
			return convolution_layer::layer_guid;
		}

		layer_testing_schema_smart_ptr convolution_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema_smart_ptr(new convolution_layer_testing_schema());
		}

		layer_tester_cuda_smart_ptr convolution_layer_testing_schema::create_tester_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			layer_tester_cuda_smart_ptr res;

			if (output_configuration_specific.get_neuron_count() == output_configuration_specific.feature_map_count)
			{
				res = layer_tester_cuda_smart_ptr(new fully_connected_layer_tester_cuda());
			}
			else
			{
				switch (output_configuration_specific.dimension_sizes.size())
				{
				case 1:
					if (cuda_config->get_compute_capability() >= 300)
						res = layer_tester_cuda_smart_ptr(new convolution_1d_layer_tester_cuda_kepler());
					else
						res = layer_tester_cuda_smart_ptr(new convolution_1d_layer_tester_cuda_fermi());
					break;
				case 2:
					if (cuda_config->get_compute_capability() >= 300)
						res = layer_tester_cuda_smart_ptr(new convolution_2d_layer_tester_cuda_kepler());
					else
						res = layer_tester_cuda_smart_ptr(new convolution_2d_layer_tester_cuda_fermi());
					break;
				case 3:
					if (cuda_config->get_compute_capability() >= 300)
						res = layer_tester_cuda_smart_ptr(new convolution_3d_layer_tester_cuda_kepler());
					else
						res = layer_tester_cuda_smart_ptr(new convolution_3d_layer_tester_cuda_fermi());
					break;
				default:
					throw neural_network_exception((boost::format("No CUDA tester for the convolutional layer of %1% dimensions") % output_configuration_specific.dimension_sizes.size()).str());
					break;
				}
			}

			return res;
		}
	}
}
