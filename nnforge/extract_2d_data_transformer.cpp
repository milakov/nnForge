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

#include "extract_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	extract_2d_data_transformer::extract_2d_data_transformer(
		unsigned int input_window_width,
		unsigned int input_window_height,
		unsigned int output_window_width,
		unsigned int output_window_height)
		: input_window_width(input_window_width)
		, input_window_height(input_window_height)
		, output_window_width(output_window_width)
		, output_window_height(output_window_height)
	{
	}

	extract_2d_data_transformer::~extract_2d_data_transformer()
	{
	}

	void extract_2d_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("extract_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("extract_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		int window_top_left_x = (original_config.dimension_sizes[0] - input_window_width) / 2;
		int window_bottom_right_x = window_top_left_x + input_window_width;
		int window_top_left_y = (original_config.dimension_sizes[1] - input_window_height) / 2;
		int window_bottom_right_y = window_top_left_y + input_window_height;

		unsigned int original_neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		unsigned int transformed_neuron_count_per_feature_map = get_transformed_configuration(original_config).get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			cv::Mat1b original_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<unsigned char *>(static_cast<const unsigned char *>(data)) + (original_neuron_count_per_feature_map * feature_map_id));
			cv::Mat1b cropped_image = original_image.rowRange(window_top_left_y, window_bottom_right_y).colRange(window_top_left_x, window_bottom_right_x);
			cv::Mat1b dest_image(static_cast<int>(output_window_height), static_cast<int>(output_window_width), static_cast<unsigned char *>(data_transformed) + (transformed_neuron_count_per_feature_map * feature_map_id));
			cv::resize(cropped_image, dest_image, dest_image.size());
		}
	}

	layer_configuration_specific extract_2d_data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("extract_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		layer_configuration_specific res(original_config.feature_map_count);
		res.dimension_sizes.resize(2);
		res.dimension_sizes[0] = output_window_width;
		res.dimension_sizes[1] = output_window_height;

		return res;
	}

	bool extract_2d_data_transformer::is_in_place() const
	{
		return false;
	}
}
