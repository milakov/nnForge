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

#include "training_imagenet_raw_to_structured_data_transformer.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

training_imagenet_raw_to_structured_data_transformer::training_imagenet_raw_to_structured_data_transformer(
	unsigned int min_image_size,
	unsigned int max_image_size,
	unsigned int target_image_width,
	unsigned int target_image_height)
	: target_image_width(target_image_width)
	, target_image_height(target_image_height)
	, gen(nnforge::rnd::get_random_generator())
	, dist(static_cast<float>(min_image_size), static_cast<float>(max_image_size))
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

	float rescaled_image_size = dist.min();
	if (dist.max() > dist.min())
	{
		boost::lock_guard<boost::mutex> lock(gen_mutex);
		rescaled_image_size = dist(gen);
	}

	float scale = static_cast<float>(std::min(original_image.rows, original_image.cols)) / rescaled_image_size;

	unsigned int source_crop_image_width = std::min(static_cast<unsigned int>(static_cast<float>(target_image_width) * scale + 0.5F), static_cast<unsigned int>(original_image.cols));
	unsigned int source_crop_image_height = std::min(static_cast<unsigned int>(static_cast<float>(target_image_height) * scale + 0.5F), static_cast<unsigned int>(original_image.rows));

	nnforge_uniform_int_distribution<unsigned int> x_dist(0, original_image.cols - source_crop_image_width);
	nnforge_uniform_int_distribution<unsigned int> y_dist(0, original_image.rows - source_crop_image_height);

	unsigned int x = x_dist.min();
	if (x_dist.max() > x_dist.min())
	{
		boost::lock_guard<boost::mutex> lock(gen_mutex);
		x = x_dist(gen);
	}

	unsigned int y = y_dist.min();
	if (y_dist.max() > y_dist.min())
	{
		boost::lock_guard<boost::mutex> lock(gen_mutex);
		y = y_dist(gen);
	}

	cv::Mat3b source_image_crop = original_image.rowRange(y, y + source_crop_image_height).colRange(x, x + source_crop_image_height);
	cv::Mat3b target_image(target_image_height, target_image_width);
	cv::resize(source_image_crop, target_image, target_image.size());

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
