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

#include "extract_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	extract_data_transformer::extract_data_transformer(
		const std::vector<unsigned int>& input_window_sizes,
		const std::vector<unsigned int>& output_window_sizes)
		: input_window_sizes(input_window_sizes)
		, output_window_sizes(output_window_sizes)
	{
		if (input_window_sizes.size() != output_window_sizes.size())
			throw neural_network_exception((boost::format("extract_data_transformer is created with different dimensions: %1% and %2%") % input_window_sizes.size() % output_window_sizes.size()).str());
	}

	extract_data_transformer::~extract_data_transformer()
	{
	}

	void extract_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (input_window_sizes == output_window_sizes)
		{
			const std::vector<unsigned int>& dimension_sizes = original_config.dimension_sizes;

			if (dimension_sizes.size() != input_window_sizes.size())
				throw neural_network_exception((boost::format("extract_data_transformer is created with %1%-dimensional rotations, data has %2% dimensions") % input_window_sizes.size() % dimension_sizes.size()).str());

			size_t elem_size = neuron_data_type::get_input_size(type);

			std::vector<unsigned int> src_offset_list;
			for(unsigned int i = 0; i < dimension_sizes.size(); ++i)
				src_offset_list.push_back((dimension_sizes[i] - output_window_sizes[i]) / 2);

			std::vector<unsigned int> dst_pos_list(dimension_sizes.size(), 0);

			const unsigned char * src_begin = (const unsigned char *)data;
			unsigned char * dst = (unsigned char *)data_transformed;

			for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
			{
				while (true)
				{
					unsigned int offset = dst_pos_list.back() + src_offset_list.back();
					for(int i = dimension_sizes.size() - 2; i >= 0; --i)
						offset = offset * dimension_sizes[i] + dst_pos_list[i] + src_offset_list[i];

					memcpy(dst, src_begin + offset * elem_size, output_window_sizes[0] * elem_size);
					dst += output_window_sizes[0] * elem_size;

					bool inc = false;
					for(int i = 1; i < output_window_sizes.size(); ++i)
					{
						dst_pos_list[i]++;
						if (dst_pos_list[i] < output_window_sizes[i])
						{
							inc = true;
							break;
						}
						else
							dst_pos_list[i] = 0;
					}
					if (!inc)
						break;
				}

				src_begin += original_config.get_neuron_count_per_feature_map() * elem_size;
			}
		}
		else
		{
			if (type != neuron_data_type::type_byte)
				throw neural_network_exception("Resizing extract_data_transformer is implemented for data stored as bytes only");

			if (original_config.dimension_sizes.size() != 2)
				throw neural_network_exception((boost::format("Resizing extract_data_transformer is processing 2D data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

			int window_top_left_x = (original_config.dimension_sizes[0] - input_window_sizes[0]) / 2;
			int window_bottom_right_x = window_top_left_x + input_window_sizes[0];
			int window_top_left_y = (original_config.dimension_sizes[1] - input_window_sizes[1]) / 2;
			int window_bottom_right_y = window_top_left_y + input_window_sizes[1];

			unsigned int original_neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
			unsigned int transformed_neuron_count_per_feature_map = get_transformed_configuration(original_config).get_neuron_count_per_feature_map();
			for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
			{
				cv::Mat1b original_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<unsigned char *>(static_cast<const unsigned char *>(data)) + (original_neuron_count_per_feature_map * feature_map_id));
				cv::Mat1b cropped_image = original_image.rowRange(window_top_left_y, window_bottom_right_y).colRange(window_top_left_x, window_bottom_right_x);
				cv::Mat1b dest_image(static_cast<int>(output_window_sizes[1]), static_cast<int>(output_window_sizes[0]), static_cast<unsigned char *>(data_transformed) + (transformed_neuron_count_per_feature_map * feature_map_id));
				cv::resize(cropped_image, dest_image, dest_image.size());
			}
		}
	}

	layer_configuration_specific extract_data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		layer_configuration_specific res(original_config.feature_map_count);
		res.dimension_sizes = output_window_sizes;

		return res;
	}

	bool extract_data_transformer::is_in_place() const
	{
		return false;
	}
}
