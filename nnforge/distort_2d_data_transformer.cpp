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

#include "distort_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	distort_2d_data_transformer::distort_2d_data_transformer(
		bool is_same_sequence_from_reset,
		float max_absolute_rotation_angle_in_degrees,
		float max_scale_factor,
		float max_absolute_shift_x,
		float max_absolute_shift_y,
		float max_contrast_factor,
		float max_absolute_brightness_shift,
		bool flip_around_x_axis_allowed,
		bool flip_around_y_axis_allowed)
		: is_same_sequence_from_reset(is_same_sequence_from_reset)
	{
		if (!is_same_sequence_from_reset)
			generator = rnd::get_random_generator();

		rotate_angle_distribution = std::tr1::uniform_real<float>(-max_absolute_rotation_angle_in_degrees, max_absolute_rotation_angle_in_degrees);
		scale_distribution = std::tr1::uniform_real<float>(1.0F / max_scale_factor, max_scale_factor);
		shift_x_distribution = std::tr1::uniform_real<float>(-max_absolute_shift_x, max_absolute_shift_x);
		shift_y_distribution = std::tr1::uniform_real<float>(-max_absolute_shift_y, max_absolute_shift_y);
		contrast_distribution = std::tr1::uniform_real<float>(1.0F / max_contrast_factor, max_contrast_factor);
		brightness_shift_distribution = std::tr1::uniform_real<float>(-max_absolute_brightness_shift, max_absolute_brightness_shift);
		flip_around_x_distribution = std::tr1::uniform_int<int>(0, flip_around_x_axis_allowed ? 1 : 0);
		flip_around_y_distribution = std::tr1::uniform_int<int>(0, flip_around_y_axis_allowed ? 1 : 0);
	}

	distort_2d_data_transformer::~distort_2d_data_transformer()
	{
	}

	void distort_2d_data_transformer::reset()
	{
		if (is_same_sequence_from_reset)
			generator = rnd::get_random_generator(48576435);
	}

	void distort_2d_data_transformer::transform(
		const void * input_data,
		void * output_data,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("distort_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("distort_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		if (original_config.feature_map_count != 1)
			throw neural_network_exception("distort_2d_data_transformer is implemented for 1 feature map data only");

		cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(output_data));

		float rotation_angle = rotate_angle_distribution(generator);
		float scale = scale_distribution(generator);
		float shift_x = shift_x_distribution(generator);
		float shift_y = shift_y_distribution(generator);
		float contrast = contrast_distribution(generator);
		float brightness_shift = brightness_shift_distribution(generator) * 255.0F;
		bool flip_around_x_axis = (flip_around_x_distribution(generator) == 1);
		bool flip_around_y_axis = (flip_around_y_distribution(generator) == 1);

		data_transformer_util::change_brightness_and_contrast(
			image,
			contrast,
			brightness_shift);

		data_transformer_util::rotate_scale_shift(
			image,
			cv::Point2f(static_cast<float>(image.cols) * 0.5F, static_cast<float>(image.rows) * 0.5F),
			rotation_angle,
			scale,
			shift_x,
			shift_y);

		data_transformer_util::flip(
			image,
			flip_around_x_axis,
			flip_around_y_axis);
	}
}
