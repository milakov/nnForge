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

#include "sparse_convolution_layer_updater_schema.h"

#include "../sparse_convolution_layer.h"
#include "../neural_network_exception.h"
#include "sparse_fully_connected_1x1_layer_updater_cuda.h"
#include "sparse_fully_connected_layer_updater_cuda.h"
#include "sparse_convolution_layer_updater_schema_helper_cuda.h"
#include "sparse_1x1_layer_updater_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		layer_updater_schema::ptr sparse_convolution_layer_updater_schema::create_specific() const
		{
			return layer_updater_schema::ptr(new sparse_convolution_layer_updater_schema());
		}

		std::string sparse_convolution_layer_updater_schema::get_type_name() const
		{
			return sparse_convolution_layer::layer_type_name;
		}

		layer_updater_cuda::ptr sparse_convolution_layer_updater_schema::create_updater_specific(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			std::shared_ptr<const sparse_convolution_layer> layer_derived = std::dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			bool zero_padding = (layer_derived->left_zero_padding == std::vector<unsigned int>(layer_derived->left_zero_padding.size(), 0))
				&& (layer_derived->right_zero_padding == std::vector<unsigned int>(layer_derived->right_zero_padding.size(), 0));
			bool unit_stride = (layer_derived->strides == std::vector<unsigned int>(layer_derived->strides.size(), 1));
			bool single_output = (output_configuration_specific.get_neuron_count() == output_configuration_specific.feature_map_count);
			bool fully_connected = single_output & unit_stride;
			bool window1x1 = (layer_derived->window_sizes == std::vector<unsigned int>(layer_derived->window_sizes.size(), 1));

			if (zero_padding)
			{
				if (fully_connected)
				{
					if (window1x1)
						return layer_updater_cuda::ptr(new sparse_fully_connected_1x1_layer_updater_cuda());
					else
						return layer_updater_cuda::ptr(new sparse_fully_connected_layer_updater_cuda());
				}
				else
				{
					if (window1x1)
						return layer_updater_cuda::ptr(new sparse_1x1_layer_updater_cuda());
				}
			}

			if (unit_stride)
				return sparse_convolution_layer_updater_schema_helper_cuda::create_updater_specific(input_configuration_specific_list[0], output_configuration_specific);

			throw neural_network_exception("There is no sparse_convolution_layer tester implemented for non-unit stride and non-unit window");
		}
	}
}
