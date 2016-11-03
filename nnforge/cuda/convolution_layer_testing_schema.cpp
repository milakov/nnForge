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

#include "convolution_layer_testing_schema.h"

#include "../convolution_layer.h"
#include "../neural_network_exception.h"
#include "fully_connected_layer_tester_cuda.h"
#include "convolution_layer_tester_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
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

			std::shared_ptr<const convolution_layer> layer_derived = std::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			bool no_padding = (layer_derived->left_zero_padding == std::vector<unsigned int>(layer_derived->left_zero_padding.size(), 0))
				&& (layer_derived->right_zero_padding == std::vector<unsigned int>(layer_derived->right_zero_padding.size(), 0));

			bool collapse_input_exactly = (input_configuration_specific_list[0].dimension_sizes == layer_derived->window_sizes);

			if (no_padding && collapse_input_exactly)
			{
				res = layer_tester_cuda::ptr(new fully_connected_layer_tester_cuda());
			}
			else if (output_configuration_specific.dimension_sizes.size() <= 3)
			{
				res = layer_tester_cuda::ptr(new convolution_layer_tester_cuda());
			}
			else
			{
				throw neural_network_exception((boost::format("No CUDA tester for the convolutional layer of %1% dimensions") % output_configuration_specific.dimension_sizes.size()).str());
			}

			return res;
		}
	}
}
