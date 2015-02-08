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

#pragma once

#include <opencv2/core/core.hpp>

namespace nnforge
{
	class data_transformer_util
	{
	public:
		static void stretch_rotate_scale_shift_perspective(
			cv::Mat image,
			cv::Point2f rotation_center,
			float angle_in_degrees,
			float scale, // 1.0F is neutral
			float shift_x,
			float shift_y,
			float stretch, // 1.0F is neutral
			float stretch_angle_in_degrees,
			float perspective_view_distance, // std::numeric_limits<float>::max() is neutral
			float perspective_view_angle,
			unsigned char border_value = 128);

		static void stretch_rotate_scale_shift_perspective(
			cv::Mat dest_image,
			const cv::Mat image,
			cv::Point2f rotation_center,
			float angle_in_degrees,
			float scale, // 1.0F is neutral
			float shift_x,
			float shift_y,
			float stretch, // 1.0F is neutral
			float stretch_angle_in_degrees,
			float perspective_view_distance, // std::numeric_limits<float>::max() is neutral
			float perspective_view_angle,
			unsigned char border_value = 128);

		// contrast: relative multiplication, about 1.0
		// brightness: change in luminocity for the middle lightness
		static void change_brightness_and_contrast(
			cv::Mat image,
			float contrast,
			float brightness);

		static void flip(
			cv::Mat image,
			bool flip_around_x_axis,
			bool flip_around_y_axis);

		static void rotate_band(
			cv::Mat image,
			int shift_x_to_left,
			int shift_y_to_top);

	private:
		data_transformer_util();
		~data_transformer_util();
	};
}
