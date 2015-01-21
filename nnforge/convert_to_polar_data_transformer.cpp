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

#include "convert_to_polar_data_transformer.h"

#include "neural_network_exception.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	const float convert_to_polar_data_transformer::pi = atan(1.0F) * 4.0F;

	convert_to_polar_data_transformer::convert_to_polar_data_transformer(
		const std::vector<unsigned int>& input_window_sizes,
		const std::vector<unsigned int>& output_window_sizes,
		float start_angle,
		float step_angle,
		float input_radius,
		unsigned char border_value)
		: input_window_sizes(input_window_sizes)
		, output_window_sizes(output_window_sizes)
		, start_angle(start_angle)
		, step_angle(step_angle)
		, input_radius(input_radius)
		, border_value(border_value)
	{
		if (input_window_sizes.size() != output_window_sizes.size())
			throw neural_network_exception((boost::format("convert_to_polar_data_transformer is created with different dimensions: %1% and %2%") % input_window_sizes.size() % output_window_sizes.size()).str());

		if (input_window_sizes.size() != 2)
			throw neural_network_exception((boost::format("convert_to_polar_data_transformer is created with wrong dimension count: %1%") % input_window_sizes.size()).str());

		map_x.create(output_window_sizes[1], output_window_sizes[0]);
		map_y.create(output_window_sizes[1], output_window_sizes[0]);
		float step_theta = step_angle * (pi * (2.0F / 360.0F));
		float start_theta = start_angle * (pi * (2.0F / 360.0F));
		float step_r = input_radius / static_cast<float>(output_window_sizes[1]);
		float start_r = step_r;
		float mid_input_x = static_cast<float>(input_window_sizes[0] - 1) * 0.5F;
		float mid_input_y = static_cast<float>(input_window_sizes[1] - 1) * 0.5F;
		for(int outut_y = 0; outut_y < map_x.rows; ++outut_y)
		{
			float r = start_r + static_cast<float>(map_x.rows - 1 - outut_y) * step_r;
			for(int ouput_x = 0; ouput_x < map_x.cols; ++ouput_x)
			{
				float theta = start_theta + static_cast<float>(ouput_x) * step_theta;
				float input_x = cosf(theta) * r + mid_input_x;
				float input_y = sinf(theta) * r + mid_input_y;
				map_x.at<float>(outut_y, ouput_x) = input_x;
				map_y.at<float>(outut_y, ouput_x) = input_y;
			}
		}
	}

	convert_to_polar_data_transformer::~convert_to_polar_data_transformer()
	{
	}

	void convert_to_polar_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("convert_to_polar_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("convert_to_polar_data_transformer is processing 2D data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		if (original_config.dimension_sizes != input_window_sizes)
			throw neural_network_exception("convert_to_polar_data_transformer: input window size mismatch between creation and actual transform");

		unsigned int original_neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		unsigned int transformed_neuron_count_per_feature_map = get_transformed_configuration(original_config).get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			cv::Mat1b original_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<unsigned char *>(static_cast<const unsigned char *>(data)) + (original_neuron_count_per_feature_map * feature_map_id));
			cv::Mat1b dest_image(static_cast<int>(output_window_sizes[1]), static_cast<int>(output_window_sizes[0]), static_cast<unsigned char *>(data_transformed) + (transformed_neuron_count_per_feature_map * feature_map_id));

			// Should try INTER_CUBIC and INTER_LANCZOS4 as well
			cv::remap(original_image, dest_image, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
	}

	layer_configuration_specific convert_to_polar_data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		layer_configuration_specific res(original_config.feature_map_count);
		res.dimension_sizes = output_window_sizes;

		return res;
	}

	bool convert_to_polar_data_transformer::is_in_place() const
	{
		return false;
	}

 	bool convert_to_polar_data_transformer::is_deterministic() const
	{
		return true;
	}
}
