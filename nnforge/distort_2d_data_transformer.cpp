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
		float max_stretch_factor)
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
	}

	distort_2d_data_transformer::~distort_2d_data_transformer()
	{
	}

	void distort_2d_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("distort_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("distort_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		float rotation_angle = rotate_angle_distribution.min();
		if (rotate_angle_distribution.max() > rotate_angle_distribution.min())
			rotation_angle = rotate_angle_distribution(generator);
		float scale = scale_distribution.min();
		if (scale_distribution.max() > scale_distribution.min())
			scale = scale_distribution(generator);
		float shift_x = shift_x_distribution.min();
		if (shift_x_distribution.max() > shift_x_distribution.min())
			shift_x = shift_x_distribution(generator);
		float shift_y = shift_y_distribution.min();
		if (shift_y_distribution.max() > shift_y_distribution.min())
			shift_y = shift_y_distribution(generator);
		bool flip_around_x_axis = (flip_around_x_distribution.min() == 1);
		if (flip_around_x_distribution.max() > flip_around_x_distribution.min())
			flip_around_x_axis = (flip_around_x_distribution(generator) == 1);
		bool flip_around_y_axis = (flip_around_y_distribution.min() == 1);
		if (flip_around_y_distribution.max() > flip_around_y_distribution.min())
			flip_around_y_axis = (flip_around_y_distribution(generator) == 1);
		float stretch = stretch_distribution.min();
		if (stretch_distribution.max() > stretch_distribution.min())
			stretch = stretch_distribution(generator);
		float stretch_angle = stretch_angle_distribution.min();
		if (stretch_angle_distribution.max() > stretch_angle_distribution.min())
			stretch_angle = stretch_angle_distribution(generator);

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(data_transformed) + (neuron_count_per_feature_map * feature_map_id));

			data_transformer_util::rotate_scale_shift(
				image,
				cv::Point2f(static_cast<float>(image.cols) * 0.5F, static_cast<float>(image.rows) * 0.5F),
				rotation_angle,
				scale,
				shift_x,
				shift_y,
				stretch,
				stretch_angle);

			data_transformer_util::flip(
				image,
				flip_around_x_axis,
				flip_around_y_axis);
		}
	}

 	bool distort_2d_data_transformer::is_deterministic() const
	{
		return false;
	}
}
