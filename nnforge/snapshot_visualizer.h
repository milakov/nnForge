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

#pragma once

#include "layer_configuration_specific_snapshot.h"
#include "layer_data.h"
#include "layer_data_configuration.h"

#include <vector>
#include <opencv2/core/core.hpp>

namespace nnforge
{
	class snapshot_visualizer
	{
	public:
		static void save_snapshot_video(
			const std::vector<layer_configuration_specific_snapshot_smart_ptr>& snapshot,
			const char * file_path,
			unsigned int fps);

		static void save_ann_snapshot(
			const layer_data_list& data,
			const std::vector<layer_data_configuration_list>& layer_data_configuration_list_list,
			const char * file_path);

		static void save_2d_snapshot(
			const layer_configuration_specific_snapshot& snapshot,
			const char * file_path,
			bool is_rgb,
			bool should_normalize,
			unsigned int scale,
			const std::vector<unsigned int>& snapshot_data_dimension_list);

		static void save_3d_snapshot(
			const layer_configuration_specific_snapshot& snapshot,
			const char * file_path,
			bool is_rgb,
			bool should_normalize,
			unsigned int fps,
			unsigned int scale,
			const std::vector<unsigned int>& snapshot_data_dimension_list);

	private:
		struct normalize_pixel_helper
		{
			normalize_pixel_helper(
				float addition,
				float multiplication);

			unsigned char operator()(float x);

		private:
			float addition;
			float multiplication;
		};

		struct color_normalize_pixel_helper
		{
			color_normalize_pixel_helper(
				float addition,
				float multiplication);

			cv::Vec3b operator()(float x);

		private:
			float addition;
			float multiplication;
		};

		snapshot_visualizer();
	};
}
