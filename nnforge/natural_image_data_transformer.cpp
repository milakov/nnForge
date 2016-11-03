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

#include "natural_image_data_transformer.h"

#include "neural_network_exception.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	natural_image_data_transformer::natural_image_data_transformer(
		float brightness,
		float contrast,
		float saturation,
		float lighting)
		: generator(rnd::get_random_generator())
		, apply_brightness_distribution(brightness > 0.0F)
		, apply_contrast_distribution(contrast > 0.0F)
		, apply_saturation_distribution(saturation > 0.0F)
		, apply_lighting(lighting > 0.0F)
		, lighting_1st_eigen_alpha_distribution(0.0F, (lighting > 0.0F ? lighting : 1.0F) * 0.2175F)
		, lighting_2nd_eigen_alpha_distribution(0.0F, (lighting > 0.0F ? lighting : 1.0F) * 0.0188F)
		, lighting_3rd_eigen_alpha_distribution(0.0F, (lighting > 0.0F ? lighting : 1.0F) * 0.0045F)
	{
		if (apply_brightness_distribution)
			brightness_distribution = std::uniform_real_distribution<float>(std::uniform_real_distribution<float>(1.0F - brightness, 1.0F + brightness));
		if (apply_contrast_distribution)
			contrast_distribution = std::uniform_real_distribution<float>(std::uniform_real_distribution<float>(1.0F - contrast, 1.0F + contrast));
		if (apply_saturation_distribution)
			saturation_distribution = std::uniform_real_distribution<float>(std::uniform_real_distribution<float>(1.0F - saturation, 1.0F + saturation));
	}

	void natural_image_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.feature_map_count != 3)
			throw neural_network_exception((boost::format("natural_image_data_transformer is provided with %1% feature maps while it can work with RGB data only") % original_config.feature_map_count).str());

		float alpha_brightness;
		float alpha_contrast;
		float alpha_saturation;
		std::vector<augmentation_type> augmentations;
		float alpha_lighting_1st_eigen;
		float alpha_lighting_2nd_eigen;
		float alpha_lighting_3rd_eigen;
		{
			std::lock_guard<std::mutex> lock(gen_mutex);

			alpha_brightness = 1.0F;
			if (apply_brightness_distribution)
				alpha_brightness = brightness_distribution(generator);

			alpha_contrast = 1.0F;
			if (apply_contrast_distribution)
				alpha_contrast = contrast_distribution(generator);

			alpha_saturation = 1.0F;
			if (apply_saturation_distribution)
				alpha_saturation = saturation_distribution(generator);

			if (alpha_brightness != 1.0F)
				augmentations.push_back(augmentation_brightness);
			if (alpha_contrast != 1.0F)
				augmentations.push_back(augmentation_contrast);
			if (alpha_saturation != 1.0F)
				augmentations.push_back(augmentation_saturation);
			for(int i = static_cast<int>(augmentations.size()) - 1; i > 0; --i)
			{
				std::uniform_int_distribution<int> dist(0, i);
				int elem_id = dist(generator);
				std::swap(augmentations[elem_id], augmentations[i]);
			}

			if (apply_lighting)
			{
				alpha_lighting_1st_eigen = lighting_1st_eigen_alpha_distribution(generator);
				alpha_lighting_2nd_eigen = lighting_2nd_eigen_alpha_distribution(generator);
				alpha_lighting_3rd_eigen = lighting_3rd_eigen_alpha_distribution(generator);
			}
		}

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		const float * src_data = data;
		float * dst_data = data_transformed;
		for(std::vector<augmentation_type>::const_iterator it = augmentations.begin(); it != augmentations.end(); ++it)
		{
			switch (*it)
			{
			case augmentation_brightness:
				{
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map) * 3; ++i)
						dst_data[i] = src_data[i] * alpha_brightness;
				}
				break;
			case augmentation_contrast:
				{
					const float * src_data_red = src_data;
					const float * src_data_green = src_data + neuron_count_per_feature_map;
					const float * src_data_blue = src_data + neuron_count_per_feature_map * 2;

					double sum_red = 0.0;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						sum_red += src_data_red[i];
					double sum_green = 0.0;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						sum_green += src_data_green[i];
					double sum_blue = 0.0;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						sum_blue += src_data_blue[i];

					float avg = (static_cast<float>(sum_red) * 0.299F + static_cast<float>(sum_green) * 0.587F  + static_cast<float>(sum_blue) * 0.114F) / static_cast<float>(neuron_count_per_feature_map);
					float avg_with_alpha = avg * (1.0F - alpha_contrast);

					float * dst_data_red = dst_data;
					float * dst_data_green = dst_data + neuron_count_per_feature_map;
					float * dst_data_blue = dst_data + neuron_count_per_feature_map * 2;

					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						dst_data_red[i] = src_data_red[i] * alpha_contrast + avg_with_alpha;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						dst_data_green[i] = src_data_green[i] * alpha_contrast + avg_with_alpha;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
						dst_data_blue[i] = src_data_blue[i] * alpha_contrast + avg_with_alpha;
				}
				break;
			case augmentation_saturation:
				{
					float gray_alpha = 1.0F - alpha_saturation;
					for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
					{
						float red = src_data[i];
						float green = src_data[i + neuron_count_per_feature_map];
						float blue = src_data[i + neuron_count_per_feature_map * 2];
						float gray = 0.299F * red + 0.587F * green + 0.114F * blue;
						float gray_with_alpha = gray * gray_alpha;
						dst_data[i] = red * alpha_saturation + gray_with_alpha;
						dst_data[i + neuron_count_per_feature_map] = green * alpha_saturation + gray_with_alpha;
						dst_data[i + neuron_count_per_feature_map * 2] = blue * alpha_saturation + gray_with_alpha;
					}
				}
				break;
			}
			src_data = dst_data;
		}

		if (apply_lighting)
		{
			float red_shift = -0.5675F * alpha_lighting_1st_eigen + 0.7192F * alpha_lighting_2nd_eigen + 0.4009F * alpha_lighting_3rd_eigen;
			float green_shift = -0.5808F * alpha_lighting_1st_eigen + (-0.0045F) * alpha_lighting_2nd_eigen + (-0.8140F) * alpha_lighting_3rd_eigen;
			float blue_shift = -0.5836F * alpha_lighting_1st_eigen + (-0.6948F) * alpha_lighting_2nd_eigen + 0.4203F * alpha_lighting_3rd_eigen;

			const float * src_data_red = src_data;
			const float * src_data_green = src_data + neuron_count_per_feature_map;
			const float * src_data_blue = src_data + neuron_count_per_feature_map * 2;

			float * dst_data_red = dst_data;
			float * dst_data_green = dst_data + neuron_count_per_feature_map;
			float * dst_data_blue = dst_data + neuron_count_per_feature_map * 2;

			for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
				dst_data_red[i] = src_data_red[i] + red_shift;
			for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
				dst_data_green[i] = src_data_green[i] + green_shift;
			for(int i = 0; i < static_cast<int>(neuron_count_per_feature_map); ++i)
				dst_data_blue[i] = src_data_blue[i] + blue_shift;

			src_data = dst_data;
		}

		if (src_data != dst_data)
			memcpy(dst_data, src_data, original_config.get_neuron_count() * sizeof(float));
	}
}
