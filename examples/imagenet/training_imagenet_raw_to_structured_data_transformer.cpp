/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "training_imagenet_raw_to_structured_data_transformer.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <nnforge/elastic_deformation_2d_data_transformer.h>

training_imagenet_raw_to_structured_data_transformer::training_imagenet_raw_to_structured_data_transformer(
	float min_relative_target_area,
	float max_relative_target_area,
	unsigned int target_image_width,
	unsigned int target_image_height,
	float max_aspect_ratio_change,
	float min_elastic_deformation_intensity,
	float max_elastic_deformation_intensity,
	float min_elastic_deformation_smoothness,
	float max_elastic_deformation_smoothness)
	: target_image_width(target_image_width)
	, target_image_height(target_image_height)
	, gen(nnforge::rnd::get_random_generator())
	, dist_relative_target_area(min_relative_target_area, max_relative_target_area)
	, dist_log_aspect_ratio(-logf(max_aspect_ratio_change), logf(max_aspect_ratio_change))
	, dist_alpha(min_elastic_deformation_intensity * static_cast<float>(std::min(target_image_width, target_image_height)), max_elastic_deformation_intensity * static_cast<float>(std::min(target_image_width, target_image_height)))
	, dist_sigma(min_elastic_deformation_smoothness * static_cast<float>(std::min(target_image_width, target_image_height)), max_elastic_deformation_smoothness * static_cast<float>(std::min(target_image_width, target_image_height)))
	, displacement_distribution(std::uniform_real_distribution<float>(-1.0F, 1.0F))
{
}

training_imagenet_raw_to_structured_data_transformer::~training_imagenet_raw_to_structured_data_transformer()
{
}

void training_imagenet_raw_to_structured_data_transformer::transform(
	unsigned int sample_id,
	const std::vector<unsigned char>& raw_data,
	float * structured_data)
{
	cv::Mat3b original_image = cv::imdecode(raw_data, CV_LOAD_IMAGE_COLOR);

	// Defaults to center crop
	unsigned int source_crop_image_width = std::min(original_image.rows, original_image.cols);
	unsigned int source_crop_image_height = source_crop_image_width;
	unsigned int x = (original_image.cols - source_crop_image_width) / 2;
	unsigned int y = (original_image.rows - source_crop_image_height) / 2;

	float alpha;
	float sigma;
	int ksize;
	{
		std::lock_guard<std::mutex> lock(gen_mutex);

		alpha = dist_alpha.min();
		if (dist_alpha.max() > dist_alpha.min())
			alpha = dist_alpha(gen);
		sigma = dist_sigma.min();
		if (dist_sigma.max() > dist_sigma.min())
			sigma = dist_sigma(gen);
		ksize = static_cast<int>((sigma - 0.8F) * 3.0F + 1.0F) * 2 + 1;

		for(int attempt = 0; attempt < 100; ++attempt)
		{
			float local_area = static_cast<float>(original_image.rows * original_image.cols);
			float relative_target_area = dist_relative_target_area.min();
			if (dist_relative_target_area.max() > dist_relative_target_area.min())
				relative_target_area = dist_relative_target_area(gen);
			float target_area = local_area * relative_target_area;
			float aspect_ratio = expf(dist_log_aspect_ratio(gen));

			unsigned int new_source_crop_image_width = std::max(static_cast<unsigned int>(sqrtf(target_area * aspect_ratio) + 0.5F), 1U);
			unsigned int new_source_crop_image_height = std::max(static_cast<unsigned int>(sqrtf(target_area / aspect_ratio) + 0.5F), 1U);

			if ((new_source_crop_image_width < static_cast<unsigned int>(original_image.cols)) && (new_source_crop_image_height < static_cast<unsigned int>(original_image.rows)))
			{
				source_crop_image_width = new_source_crop_image_width;
				source_crop_image_height = new_source_crop_image_height;
				std::uniform_int_distribution<unsigned int> x_dist(0, original_image.cols - source_crop_image_width);
				std::uniform_int_distribution<unsigned int> y_dist(0, original_image.rows - source_crop_image_height);
				x = x_dist.min();
				if (x_dist.max() > x_dist.min())
					x = x_dist(gen);
				y = y_dist.min();
				if (y_dist.max() > y_dist.min())
					y = y_dist(gen);

				break;
			}
		}
	}

	cv::Mat3b target_image(target_image_height, target_image_width);
	if (alpha > 0.0F)
	{
		cv::Mat1f x_disp(target_image_height, target_image_width);
		cv::Mat1f y_disp(target_image_height, target_image_width);
		{
			std::lock_guard<std::mutex> lock(gen_mutex);

			for(int row_id = 0; row_id < x_disp.rows; ++row_id)
			{
				float * row_ptr = x_disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < x_disp.cols; ++column_id)
					row_ptr[column_id] = displacement_distribution(gen);
			}

			for(int row_id = 0; row_id < y_disp.rows; ++row_id)
			{
				float * row_ptr = y_disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < y_disp.cols; ++column_id)
					row_ptr[column_id] = displacement_distribution(gen);
			}
		}

		{
			nnforge::elastic_deformation_2d_data_transformer::smooth(x_disp, ksize, sigma, alpha, true, static_cast<float>(x), static_cast<float>(source_crop_image_width) / static_cast<float>(target_image_width));
			auto minmax_it = std::minmax_element(x_disp.begin(), x_disp.end());
			float min_value = *minmax_it.first;
			float max_value = *minmax_it.second;
			float peak_val = static_cast<float>(original_image.cols - 1);
			if ((min_value < 0.0F) || (max_value > peak_val))
			{
				float size = max_value - min_value;
				float mult = 1.0F;
				if (size > peak_val)
				{
					mult = peak_val / size;
					min_value *= mult;
					max_value *= mult;
				}
				float add = 0.0F;
				if (min_value < 0.0F)
					add = -min_value;
				if (max_value > peak_val)
					add = peak_val - max_value;

				std::for_each(x_disp.begin(), x_disp.end(), [mult, add] (float& x) { x = x * mult + add; });
			}
		}

		{
			nnforge::elastic_deformation_2d_data_transformer::smooth(y_disp, ksize, sigma, alpha, false, static_cast<float>(y), static_cast<float>(source_crop_image_height) / static_cast<float>(target_image_height));
			auto minmax_it = std::minmax_element(y_disp.begin(), y_disp.end());
			float min_value = *minmax_it.first;
			float max_value = *minmax_it.second;
			float peak_val = static_cast<float>(original_image.rows - 1);
			if ((min_value < 0.0F) || (max_value > peak_val))
			{
				float size = max_value - min_value;
				float mult = 1.0F;
				if (size > peak_val)
				{
					mult = peak_val / size;
					min_value *= mult;
					max_value *= mult;
				}
				float add = 0.0F;
				if (min_value < 0.0F)
					add = -min_value;
				if (max_value > peak_val)
					add = peak_val - max_value;

				std::for_each(y_disp.begin(), y_disp.end(), [mult, add] (float& x) { x = x * mult + add; });
			}
		}

		cv::remap(original_image, target_image, x_disp, y_disp, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(127.0F, 127.0F, 127.0F));
	}
	else
	{
		cv::Mat3b source_image_crop = original_image.rowRange(y, y + source_crop_image_height).colRange(x, x + source_crop_image_width);
		cv::resize(source_image_crop, target_image, target_image.size(), 0.0, 0.0, cv::INTER_CUBIC);
	}

	float * r_dst_it = structured_data;
	float * g_dst_it = structured_data + (target_image_width * target_image_height);
	float * b_dst_it = structured_data + (target_image_width * target_image_height * 2);
	for(cv::Mat3b::const_iterator it = target_image.begin(); it != target_image.end(); ++it, ++r_dst_it, ++g_dst_it, ++b_dst_it)
	{
		*r_dst_it = static_cast<float>((*it)[2]) * (1.0F / 255.0F);
		*g_dst_it = static_cast<float>((*it)[1]) * (1.0F / 255.0F);
		*b_dst_it = static_cast<float>((*it)[0]) * (1.0F / 255.0F);
	}
}

nnforge::layer_configuration_specific training_imagenet_raw_to_structured_data_transformer::get_configuration() const
{
	nnforge::layer_configuration_specific res(3);
	res.dimension_sizes.push_back(target_image_width);
	res.dimension_sizes.push_back(target_image_height);
	return res;
}
