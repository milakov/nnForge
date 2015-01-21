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

#include "data_transformer_util.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

namespace nnforge
{
	void data_transformer_util::rotate_scale_shift(
		cv::Mat image,
		cv::Point2f rotation_center,
		float angle_in_degrees,
		float scale,
		float shift_x,
		float shift_y,
		float stretch,
		float stretch_angle_in_degrees,
		unsigned char border_value)
	{
		cv::Mat copy = image.clone();
		rotate_scale_shift(
			image,
			copy,
			rotation_center,
			angle_in_degrees,
			scale,
			shift_x,
			shift_y,
			stretch,
			stretch_angle_in_degrees,
			border_value);
	}

	void data_transformer_util::rotate_scale_shift(
		cv::Mat dest_image,
		const cv::Mat image,
		cv::Point2f rotation_center,
		float angle_in_degrees,
		float scale,
		float shift_x,
		float shift_y,
		float stretch,
		float stretch_angle_in_degrees,
		unsigned char border_value)
	{
		cv::Mat stretch_full_mat(3, 3, CV_64FC1);
		stretch_full_mat.at<double>(2, 0) = 0.0;
		stretch_full_mat.at<double>(2, 1) = 0.0;
		stretch_full_mat.at<double>(2, 2) = 1.0;
		{
			float stretch_angle_in_radians = stretch_angle_in_degrees * (2.0F * 3.14159265358979F / 360.0F);
			float x = cosf(stretch_angle_in_radians);
			float y = sinf(stretch_angle_in_radians);
			cv::Point2f src[3];
			cv::Point2f dst[3];
			src[0] = rotation_center;
			dst[0] = src[0];
			src[1] = rotation_center + cv::Point2f(x, y);
			dst[1] = rotation_center + cv::Point2f(x * stretch, y * stretch);
			src[2] = rotation_center + cv::Point2f(y, -x);
			dst[2] = src[2];
			cv::Mat stretch_mat = cv::getAffineTransform(src, dst);

			stretch_full_mat.at<double>(0, 0) = stretch_mat.at<double>(0, 0);
			stretch_full_mat.at<double>(0, 1) = stretch_mat.at<double>(0, 1);
			stretch_full_mat.at<double>(0, 2) = stretch_mat.at<double>(0, 2);
			stretch_full_mat.at<double>(1, 0) = stretch_mat.at<double>(1, 0);
			stretch_full_mat.at<double>(1, 1) = stretch_mat.at<double>(1, 1);
			stretch_full_mat.at<double>(1, 2) = stretch_mat.at<double>(1, 2);
		}

		cv::Mat rot_mat = cv::getRotationMatrix2D(
			rotation_center,
			static_cast<double>(angle_in_degrees),
			static_cast<double>(scale));

		cv::Mat stretch_and_rot_mat = rot_mat * stretch_full_mat;

		stretch_and_rot_mat.at<double>(0, 2) += static_cast<double>(shift_x);
		stretch_and_rot_mat.at<double>(1, 2) += static_cast<double>(shift_y);

		cv::warpAffine(
			image,
			dest_image,
			stretch_and_rot_mat,
			dest_image.size(),
			cv::INTER_LINEAR,
			cv::BORDER_CONSTANT,
			border_value);
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
		int shift_x_to_left,
		int shift_y_to_top)
	{
		int actual_shift_x = (shift_x_to_left % image.cols);
		if (actual_shift_x < 0)
			actual_shift_x += image.cols;
		int actual_shift_y = (shift_y_to_top % image.rows);
		if (actual_shift_y < 0)
			actual_shift_y += image.rows;
		if ((actual_shift_x == 0) && (actual_shift_y == 0))
			return;

		cv::Mat cloned_image = image.clone();

		if (actual_shift_y == 0)
		{
			cloned_image.colRange(actual_shift_x, image.cols).copyTo(image.colRange(0, image.cols - actual_shift_x));
			cloned_image.colRange(0, actual_shift_x).copyTo(image.colRange(image.cols - actual_shift_x, image.cols));
		}
		else if (actual_shift_x == 0)
		{
			cloned_image.rowRange(actual_shift_y, image.rows).copyTo(image.rowRange(0, image.rows - actual_shift_y));
			cloned_image.rowRange(0, actual_shift_y).copyTo(image.rowRange(image.rows - actual_shift_y, image.rows));
		}
		else
		{
			cloned_image.colRange(actual_shift_x, image.cols).rowRange(actual_shift_y, image.rows).copyTo(image.colRange(0, image.cols - actual_shift_x).rowRange(0, image.rows - actual_shift_y));
			cloned_image.colRange(0, actual_shift_x).rowRange(actual_shift_y, image.rows).copyTo(image.colRange(image.cols - actual_shift_x, image.cols).rowRange(0, image.rows - actual_shift_y));
			cloned_image.colRange(actual_shift_x, image.cols).rowRange(0, actual_shift_y).copyTo(image.colRange(0, image.cols - actual_shift_x).rowRange(image.rows - actual_shift_y, image.rows));
			cloned_image.colRange(0, actual_shift_x).rowRange(0, actual_shift_y).copyTo(image.colRange(image.cols - actual_shift_x, image.cols).rowRange(image.rows - actual_shift_y, image.rows));
		}
	}
}
