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

#include "distort_2d_data_sampler_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>
#include <cstring>

namespace nnforge
{
	distort_2d_data_sampler_transformer::distort_2d_data_sampler_transformer(
		const std::vector<distort_2d_data_sampler_param>& params,
		float border_value)
		: params(params)
		, border_value(border_value)
	{
	}

	distort_2d_data_sampler_transformer::distort_2d_data_sampler_transformer(
		const std::vector<float>& rotation_angle_in_degrees_list,
		const std::vector<float>& scale_list,
		const std::vector<float>& shift_right_x_list,
		const std::vector<float>& shift_down_y_list,
		const std::vector<std::pair<float, float> >& stretch_factor_and_angle_list,
		const std::vector<std::pair<float, float> >& perspective_distance_and_angle_list,
		bool flip_around_x,
		bool flip_around_y,
		float border_value)
		: border_value(border_value)
	{
		for(std::vector<float>::const_iterator it1 = rotation_angle_in_degrees_list.begin(); it1 != rotation_angle_in_degrees_list.end(); ++it1)
			for(std::vector<float>::const_iterator it2 = scale_list.begin(); it2 != scale_list.end(); ++it2)
				for(std::vector<float>::const_iterator it3 = shift_right_x_list.begin(); it3 != shift_right_x_list.end(); ++it3)
					for(std::vector<float>::const_iterator it4 = shift_down_y_list.begin(); it4 != shift_down_y_list.end(); ++it4)
						for(std::vector<std::pair<float, float> >::const_iterator it5 = stretch_factor_and_angle_list.begin(); it5 != stretch_factor_and_angle_list.end(); ++it5)
							for(std::vector<std::pair<float, float> >::const_iterator it6 = perspective_distance_and_angle_list.begin(); it6 != perspective_distance_and_angle_list.end(); ++it6)
								for(unsigned int flip_x = 0; flip_x < (flip_around_x ? 2U : 1U); ++flip_x)
									for(unsigned int flip_y = 0; flip_y < (flip_around_y ? 2U : 1U); ++flip_y)
									{
										distort_2d_data_sampler_param new_item;
										new_item.rotation_angle_in_degrees = *it1;
										new_item.scale = *it2;
										new_item.shift_right_x = *it3;
										new_item.shift_down_y = *it4;
										new_item.stretch_factor_and_angle = *it5;
										new_item.perspective_distance_and_angle = *it6;
										new_item.flip_around_x = (flip_x == 1);
										new_item.flip_around_y = (flip_y == 1);
										params.push_back(new_item);
									}
	}

	void distort_2d_data_sampler_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.dimension_sizes.size() < 2)
			throw neural_network_exception((boost::format("distort_2d_data_sampler_transformer is processing at least 2d data, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		float rotation_angle = params[sample_id].rotation_angle_in_degrees;
		float scale = params[sample_id].scale;
		float shift_x = params[sample_id].shift_right_x;
		float shift_y = params[sample_id].shift_down_y;
		float stretch_factor = params[sample_id].stretch_factor_and_angle.first;
		float stretch_angle = params[sample_id].stretch_factor_and_angle.second;
		float perspective_distance = params[sample_id].perspective_distance_and_angle.first;
		float perspective_angle = params[sample_id].perspective_distance_and_angle.second;
		bool flip_around_x = params[sample_id].flip_around_x;
		bool flip_around_y = params[sample_id].flip_around_y;

		unsigned int neuron_count_per_image = original_config.dimension_sizes[0] * original_config.dimension_sizes[1];
		unsigned int image_count = original_config.get_neuron_count() / neuron_count_per_image;
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			cv::Mat1f image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<float *>(data) + (image_id * neuron_count_per_image));
			cv::Mat1f dest_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), data_transformed + (image_id * neuron_count_per_image));

			data_transformer_util::stretch_rotate_scale_shift_perspective(
				dest_image,
				image,
				cv::Point2f(static_cast<float>(image.cols) * 0.5F, static_cast<float>(image.rows) * 0.5F),
				rotation_angle,
				scale,
				shift_x,
				shift_y,
				stretch_factor,
				stretch_angle,
				perspective_distance,
				perspective_angle,
				border_value);

			data_transformer_util::flip(
				dest_image,
				flip_around_x,
				flip_around_y);
		}
	}

	unsigned int distort_2d_data_sampler_transformer::get_sample_count() const
	{
		return static_cast<unsigned int>(params.size());
	}
}
