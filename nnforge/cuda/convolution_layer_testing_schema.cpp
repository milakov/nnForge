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

#include "convolution_layer_testing_schema.h"

#include "../convolution_layer.h"
#include "../neural_network_exception.h"
#include "fully_connected_layer_tester_cuda.h"
#include "convolution_1x1_layer_tester_cuda.h"
#include "convolution_layer_tester_cuda.h"

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

		std::string convolution_layer_testing_schema::get_type_name() const
		{
			return convolution_layer::layer_type_name;
		}

		layer_testing_schema::ptr convolution_layer_testing_schema::create_specific() const
		{
			return layer_testing_schema::ptr(new convolution_layer_testing_schema());
		}

		layer_tester_cuda::ptr convolution_layer_testing_schema::create_tester_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			layer_tester_cuda::ptr res;

			nnforge_shared_ptr<const convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const convolution_layer>(layer_schema);

			bool zero_padding = (layer_derived->left_zero_padding == std::vector<unsigned int>(layer_derived->left_zero_padding.size(), 0))
				&& (layer_derived->right_zero_padding == std::vector<unsigned int>(layer_derived->right_zero_padding.size(), 0));

			if (zero_padding && (output_configuration_specific.get_neuron_count() == output_configuration_specific.feature_map_count))
			{
				res = layer_tester_cuda::ptr(new fully_connected_layer_tester_cuda());
			}
			else if (zero_padding && (input_configuration_specific_list[0].dimension_sizes == output_configuration_specific.dimension_sizes))
			{
				res = layer_tester_cuda::ptr(new convolution_1x1_layer_tester_cuda());
			}
			else if (output_configuration_specific.dimension_sizes.size() <= 2)
			{
				res = layer_tester_cuda::ptr(new convolution_layer_tester_cuda());
			}
			else
			{
				throw neural_network_exception((boost::format("No CUDA tester for the convolutional layer of %1% dimensions for Kepler and above architectures") % output_configuration_specific.dimension_sizes.size()).str());
			}

			return res;
		}
	}
}
