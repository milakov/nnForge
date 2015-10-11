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

#include "distort_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	distort_2d_data_transformer::distort_2d_data_transformer(
		float max_absolute_rotation_angle_in_degrees,
		float max_scale_factor,
		float min_shift_right_x,
		float max_shift_right_x,
		float min_shift_down_y,
		float max_shift_down_y,
		bool flip_around_x_axis_allowed,
		bool flip_around_y_axis_allowed,
		float max_stretch_factor,
		float min_perspective_distance,
		float border_value)
		: border_value(border_value)
	{
		generator = rnd::get_random_generator();

		rotate_angle_distribution = nnforge_uniform_real_distribution<float>(-max_absolute_rotation_angle_in_degrees, max_absolute_rotation_angle_in_degrees);
		scale_distribution = nnforge_uniform_real_distribution<float>(1.0F / max_scale_factor, max_scale_factor);
		shift_x_distribution = nnforge_uniform_real_distribution<float>(min_shift_right_x, max_shift_right_x);
		shift_y_distribution = nnforge_uniform_real_distribution<float>(min_shift_down_y, max_shift_down_y);
		flip_around_x_distribution = nnforge_uniform_int_distribution<int>(0, flip_around_x_axis_allowed ? 1 : 0);
		flip_around_y_distribution = nnforge_uniform_int_distribution<int>(0, flip_around_y_axis_allowed ? 1 : 0);
		stretch_distribution = nnforge_uniform_real_distribution<float>(1.0F / max_stretch_factor, max_stretch_factor);
		stretch_angle_distribution = nnforge_uniform_real_distribution<float>(-180.0F, 180.0F);
		perspective_reverse_distance_distribution = nnforge_uniform_real_distribution<float>(0.0F, (min_perspective_distance == std::numeric_limits<float>::max()) ? 0.0F : 1.0F / min_perspective_distance);
		perspective_angle_distribution = nnforge_uniform_real_distribution<float>(-180.0F, 180.0F);
	}

	distort_2d_data_transformer::~distort_2d_data_transformer()
	{
	}

	void distort_2d_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.dimension_sizes.size() < 2)
			throw neural_network_exception((boost::format("distort_2d_data_transformer is processing at least 2d data, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		float rotation_angle = rotate_angle_distribution.min();
		float scale = scale_distribution.min();
		float shift_x = shift_x_distribution.min();
		float shift_y = shift_y_distribution.min();
		bool flip_around_x_axis = (flip_around_x_distribution.min() == 1);
		bool flip_around_y_axis = (flip_around_y_distribution.min() == 1);
		float stretch = stretch_distribution.min();
		float stretch_angle = stretch_angle_distribution.min();
		float perspective_reverse_distance = perspective_reverse_distance_distribution.min();
		float perspective_distance = std::numeric_limits<float>::max();
		float perspective_angle = perspective_angle_distribution.min();

		{
			boost::lock_guard<boost::mutex> lock(gen_stream_mutex);

			if (rotate_angle_distribution.max() > rotate_angle_distribution.min())
				rotation_angle = rotate_angle_distribution(generator);
			if (scale_distribution.max() > scale_distribution.min())
				scale = scale_distribution(generator);
			if (shift_x_distribution.max() > shift_x_distribution.min())
				shift_x = shift_x_distribution(generator);
			if (shift_y_distribution.max() > shift_y_distribution.min())
				shift_y = shift_y_distribution(generator);
			if (flip_around_x_distribution.max() > flip_around_x_distribution.min())
				flip_around_x_axis = (flip_around_x_distribution(generator) == 1);
			if (flip_around_y_distribution.max() > flip_around_y_distribution.min())
				flip_around_y_axis = (flip_around_y_distribution(generator) == 1);
			if (stretch_distribution.max() > stretch_distribution.min())
				stretch = stretch_distribution(generator);
			if (stretch_angle_distribution.max() > stretch_angle_distribution.min())
				stretch_angle = stretch_angle_distribution(generator);
			if (perspective_reverse_distance_distribution.max() > perspective_reverse_distance_distribution.min())
				perspective_reverse_distance = perspective_reverse_distance_distribution(generator);
			if (perspective_reverse_distance > 0.0F)
				perspective_distance = 1.0F / perspective_reverse_distance;
			if (perspective_angle_distribution.max() > perspective_angle_distribution.min())
				perspective_angle = perspective_angle_distribution(generator);
		}

		unsigned int neuron_count_per_image = original_config.dimension_sizes[0] * original_config.dimension_sizes[1];
		unsigned int image_count = original_config.get_neuron_count() / neuron_count_per_image;
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			cv::Mat1f dest_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), data_transformed + (image_id * neuron_count_per_image));
			cv::Mat1f image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<float *>(data) + (image_id * neuron_count_per_image));

			if ((rotation_angle != 0.0F) || (scale != 1.0F) || (shift_x != 0.0F) || (shift_y != 0.0F) || (stretch != 1.0F) || (perspective_distance != std::numeric_limits<float>::max()))
			{
				data_transformer_util::stretch_rotate_scale_shift_perspective(
					dest_image,
					image,
					cv::Point2f(static_cast<float>(image.cols) * 0.5F, static_cast<float>(image.rows) * 0.5F),
					rotation_angle,
					scale,
					shift_x,
					shift_y,
					stretch,
					stretch_angle,
					perspective_distance,
					perspective_angle,
					border_value);

				data_transformer_util::flip(
					dest_image,
					flip_around_x_axis,
					flip_around_y_axis);
			}
			else
			{
				data_transformer_util::flip(
					dest_image,
					image,
					flip_around_x_axis,
					flip_around_y_axis);
			}
		}
	}
}
