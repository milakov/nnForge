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

#include "data_transformer_util.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace nnforge
{
	void data_transformer_util::rotate_scale_shift(
		cv::Mat image,
		cv::Point2f rotation_center,
		float angle_in_degrees,
		float scale,
		float shift_x,
		float shift_y)
	{
		if ((angle_in_degrees != 0.0F) || (scale != 1.0F) || (shift_x != 0.0F) || (shift_y != 0.0F))
		{
			cv::Mat rot_mat = cv::getRotationMatrix2D(
				rotation_center,
				static_cast<double>(angle_in_degrees),
				static_cast<double>(scale));

			rot_mat.at<double>(0, 2) += static_cast<double>(shift_x);
			rot_mat.at<double>(1, 2) += static_cast<double>(shift_y);

			cv::Mat copy = image.clone();
			cv::warpAffine(
				copy,
				image,
				rot_mat,
				image.size(),
				cv::INTER_LINEAR,
				cv::BORDER_CONSTANT,
				128);
		}
	}

	void data_transformer_util::change_brightness_and_contrast(
		cv::Mat image,
		float contrast,
		float brightness)
	{
		if ((contrast != 1.0F) || (brightness != 0.0F))
		{
			image.convertTo(
				image,
				-1,
				static_cast<double>(contrast),
				static_cast<double>(brightness));
		}
	}

	void data_transformer_util::flip(
		cv::Mat image,
		bool flip_around_x_axis,
		bool flip_around_y_axis)
	{
		int flip_code;
		if (flip_around_x_axis)
		{
			if (flip_around_y_axis)
				flip_code = -1;
			else
				flip_code = 0;
		}
		else
		{
			if (flip_around_y_axis)
				flip_code = 1;
			else
				return;
		}
		
		cv::flip(image, image, flip_code);
	}

	void data_transformer_util::rotate_band(
		cv::Mat image,
		int shift_x_to_left)
	{
		unsigned int real_shift = (shift_x_to_left % image.cols);
		if (real_shift == 0)
			return;

		cv::Mat cloned_image = image.clone();
		cloned_image.colRange(real_shift, image.cols).copyTo(image.colRange(0, image.cols - real_shift));
		cloned_image.colRange(0, real_shift).copyTo(image.colRange(image.cols - real_shift, image.cols));
	}
}
