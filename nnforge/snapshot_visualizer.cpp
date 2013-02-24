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

#include "snapshot_visualizer.h"

#include "neural_network_exception.h"

#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>

#include <algorithm>
#include <numeric>

namespace nnforge
{
	void snapshot_visualizer::save_snapshot(
		const std::vector<layer_configuration_specific_snapshot_smart_ptr>& snapshot,
		const char * file_path)
	{
		std::vector<unsigned int> layer_offset_from_top;
		unsigned int layer_offset_from_top_current = 0;
		unsigned int width = 0;
		for(unsigned int i = 0; i < snapshot.size(); ++i)
		{
			const layer_configuration_specific& current_config = snapshot[i]->config;

			layer_offset_from_top.push_back(layer_offset_from_top_current);

			if (current_config.dimension_sizes.size() > 2)
				throw neural_network_exception((boost::format("Unable to save snapshot to image for dimension size %1%") % current_config.dimension_sizes.size()).str());

			unsigned int current_height = current_config.dimension_sizes.size() > 1 ? current_config.dimension_sizes[1] : 1;
			unsigned int current_width = current_config.dimension_sizes[0] * current_config.feature_map_count;

			width = std::max<unsigned int>(width, current_width);
			layer_offset_from_top_current += current_height + 1;
		}

		cv::Mat_<unsigned char> image(layer_offset_from_top_current - 1, width, 255);

		for(unsigned int i = 0; i < snapshot.size(); ++i)
		{
			const layer_configuration_specific& current_config = snapshot[i]->config;
			const std::vector<float>& current_data = snapshot[i]->data;
			unsigned int current_height = current_config.dimension_sizes.size() > 1 ? current_config.dimension_sizes[1] : 1;
			
			cv::Mat_<unsigned char> subImage = image.rowRange(layer_offset_from_top[i], layer_offset_from_top[i] + current_height);

			float min = *std::min_element(current_data.begin(), current_data.end());
			float max = *std::max_element(current_data.begin(), current_data.end());
			if (min >= max)
			{
				min = std::min<float>(min, -1.0F);
				max = std::max<float>(max, 1.0F);
			}

			float addition = -min;
			float multiplication = 255.0F / (max - min);

			normalize_pixel_helper norm(addition, multiplication);
			unsigned int feature_map_width = current_config.dimension_sizes[0];

			for(unsigned int feature_map_id = 0; feature_map_id < current_config.feature_map_count; ++feature_map_id)
			{
				cv::Mat_<unsigned char> featureMapSubImage = subImage.colRange(feature_map_width * feature_map_id, feature_map_width * (feature_map_id + 1));

				std::transform(
					current_data.begin() + ((current_height * feature_map_width) * feature_map_id),
					current_data.begin() + ((current_height * feature_map_width) * (feature_map_id + 1)),
					featureMapSubImage.begin(),
					norm);
			}
		}

		if (!cv::imwrite(file_path, image))
			throw std::runtime_error((boost::format("Error saving snapshot to %1%") % file_path).str());
	}

	void snapshot_visualizer::save_snapshot_video(
		const std::vector<layer_configuration_specific_snapshot_smart_ptr>& snapshot,
		const char * file_path,
		unsigned int fps)
	{
		std::vector<unsigned int> layer_offset_from_top;
		unsigned int layer_offset_from_top_current = 0;
		unsigned int width = 0;
		unsigned int max_depth = 0;
		std::vector<color_normalize_pixel_helper> color_normalize_pixel_helper_list;
		for(unsigned int i = 0; i < snapshot.size(); ++i)
		{
			const layer_configuration_specific& current_config = snapshot[i]->config;

			layer_offset_from_top.push_back(layer_offset_from_top_current);

			if (current_config.dimension_sizes.size() != 3)
				throw neural_network_exception((boost::format("Unable to save snapshot to video for dimension size %1%") % current_config.dimension_sizes.size()).str());

				const std::vector<float>& current_data = snapshot[i]->data;
			float min = *std::min_element(current_data.begin(), current_data.end());
			float max = *std::max_element(current_data.begin(), current_data.end());
			if (min >= max)
			{
				min = std::min<float>(min, -1.0F);
				max = std::max<float>(max, 1.0F);
			}
			float addition = -min;
			float multiplication = 255.0F / (max - min);
			color_normalize_pixel_helper_list.push_back(color_normalize_pixel_helper(addition, multiplication));

			unsigned int current_height = current_config.dimension_sizes[1];
			unsigned int current_width = current_config.dimension_sizes[0] * current_config.feature_map_count;

			width = std::max<unsigned int>(width, current_width);
			layer_offset_from_top_current += current_height + 1;
			max_depth = std::max<unsigned int>(max_depth, current_config.dimension_sizes[2]);
		}
		unsigned int height = layer_offset_from_top_current - 1;

		cv::VideoWriter video(file_path, CV_FOURCC('D','I','V','X'), fps, cv::Size(width, height), true);
		if (!video.isOpened())
			throw std::runtime_error((boost::format("Error saving video snapshot to file %1%") % file_path).str());

		for(unsigned int frame_id = 0; frame_id < max_depth; frame_id++)
		{
			cv::Mat3b image(height, width, cv::Vec3b(255, 255, 255));

			for(unsigned int i = 0; i < snapshot.size(); ++i)
			{
				const layer_configuration_specific& current_config = snapshot[i]->config;

				int current_depth = static_cast<int>(current_config.dimension_sizes[2] + frame_id) - static_cast<int>(max_depth);
				if ((current_depth < 0) || (current_depth >= static_cast<int>(current_config.dimension_sizes[2])))
					continue;

				unsigned int feature_map_width = current_config.dimension_sizes[0];
				unsigned int current_height = current_config.dimension_sizes[1];
				unsigned int depth = current_config.dimension_sizes[2];
				unsigned int data_offset = feature_map_width * current_height * current_depth;
				const std::vector<float>& current_data = snapshot[i]->data;
			
				cv::Mat3b subImage = image.rowRange(layer_offset_from_top[i], layer_offset_from_top[i] + current_height);

				for(unsigned int feature_map_id = 0; feature_map_id < current_config.feature_map_count; ++feature_map_id)
				{
					cv::Mat3b featureMapSubImage = subImage.colRange(feature_map_width * feature_map_id, feature_map_width * (feature_map_id + 1));

					std::transform(
						current_data.begin() + ((current_height * feature_map_width * depth) * feature_map_id) + data_offset,
						current_data.begin() + ((current_height * feature_map_width * depth) * feature_map_id) + data_offset + (current_height * feature_map_width),
						featureMapSubImage.begin(),
						color_normalize_pixel_helper_list[i]);
				}
			}

			video << image;
		}
	}

	snapshot_visualizer::normalize_pixel_helper::normalize_pixel_helper(
		float addition,
		float multiplication)
		: addition(addition), multiplication(multiplication)
	{
	}

	unsigned char snapshot_visualizer::normalize_pixel_helper::operator()(float x)
	{
		return static_cast<unsigned char>(std::min<float>(std::max<float>((x + addition) * multiplication, 0.0F), 255.0F));
	}

	snapshot_visualizer::color_normalize_pixel_helper::color_normalize_pixel_helper(
		float addition,
		float multiplication)
		: addition(addition), multiplication(multiplication)
	{
	}

	cv::Vec3b snapshot_visualizer::color_normalize_pixel_helper::operator()(float x)
	{
		unsigned char val = static_cast<unsigned char>(std::min<float>(std::max<float>((x + addition) * multiplication, 0.0F), 255.0F));
		return cv::Vec3b(val, val, val);
	}
}
