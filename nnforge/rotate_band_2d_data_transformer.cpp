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

#include "rotate_band_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	rotate_band_2d_data_transformer::rotate_band_2d_data_transformer(
		unsigned int max_absolute_band_rotation_x,
		unsigned int max_absolute_band_rotation_y)
	{
		generator = rnd::get_random_generator();

		rotate_band_x_distribution = std::tr1::uniform_int<int>(-static_cast<int>(max_absolute_band_rotation_x), static_cast<int>(max_absolute_band_rotation_x));
		rotate_band_y_distribution = std::tr1::uniform_int<int>(-static_cast<int>(max_absolute_band_rotation_y), static_cast<int>(max_absolute_band_rotation_y));
	}

	rotate_band_2d_data_transformer::~rotate_band_2d_data_transformer()
	{
	}

	void rotate_band_2d_data_transformer::transform(
		const void * input_data,
		void * output_data,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("rotate_band_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("rotate_band_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		if (original_config.feature_map_count != 1)
			throw neural_network_exception("rotate_band_2d_data_transformer is implemented for 1 feature map data only");

		cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(output_data));

		int rotate_band_x = rotate_band_x_distribution(generator);
		int rotate_band_y = rotate_band_y_distribution(generator);

		data_transformer_util::rotate_band(
			image,
			rotate_band_x,
			rotate_band_y);
	}
}
