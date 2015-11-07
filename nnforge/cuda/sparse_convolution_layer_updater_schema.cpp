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

#include "sparse_convolution_layer_updater_schema.h"

#include "../sparse_convolution_layer.h"
#include "../neural_network_exception.h"
#include "sparse_fully_connected_1x1_layer_updater_cuda.h"
#include "sparse_fully_connected_layer_updater_cuda.h"
#include "sparse_convolution_layer_updater_schema_helper_cuda_kepler.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		sparse_convolution_layer_updater_schema::sparse_convolution_layer_updater_schema()
		{
		}

		sparse_convolution_layer_updater_schema::~sparse_convolution_layer_updater_schema()
		{
		}

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
			layer_updater_cuda::ptr res;

			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			bool zero_padding = (layer_derived->left_zero_padding == std::vector<unsigned int>(layer_derived->left_zero_padding.size(), 0))
				&& (layer_derived->right_zero_padding == std::vector<unsigned int>(layer_derived->right_zero_padding.size(), 0));

			if (zero_padding && (output_configuration_specific.get_neuron_count() == output_configuration_specific.feature_map_count))
			{
				if (input_configuration_specific_list[0].dimension_sizes == output_configuration_specific.dimension_sizes)
				{
					res = layer_updater_cuda::ptr(new sparse_fully_connected_1x1_layer_updater_cuda());
				}
				else
				{
					res = layer_updater_cuda::ptr(new sparse_fully_connected_layer_updater_cuda());
				}
			}
			else
			{
				res = sparse_convolution_layer_updater_schema_helper_cuda_kepler::create_updater_specific(input_configuration_specific_list, output_configuration_specific);
			}

			return res;
		}
	}
}
