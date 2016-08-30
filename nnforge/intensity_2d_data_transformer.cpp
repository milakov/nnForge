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

#include "intensity_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	intensity_2d_data_transformer::intensity_2d_data_transformer(
		float max_contrast_factor,
		float max_absolute_brightness_shift)
		: apply_contrast_distribution(max_contrast_factor > 1.0F)
		, apply_brightness_shift_distribution(apply_brightness_shift_distribution != 0.0F)
	{
		generator = rnd::get_random_generator();

		contrast_distribution = nnforge_uniform_real_distribution<float>(1.0F / max_contrast_factor, max_contrast_factor + (apply_contrast_distribution ? 0.0F: 1.0F));
		brightness_shift_distribution = nnforge_uniform_real_distribution<float>(-max_absolute_brightness_shift, max_absolute_brightness_shift + (apply_brightness_shift_distribution ? 0.0F : 1.0F));
	}

	intensity_2d_data_transformer::~intensity_2d_data_transformer()
	{
	}

	void intensity_2d_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.dimension_sizes.size() < 2)
			throw neural_network_exception((boost::format("intensity_2d_data_transformer is processing at least 2d data, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		float contrast = contrast_distribution.min();
		float brightness_shift = brightness_shift_distribution.min();

		{
			boost::lock_guard<boost::mutex> lock(gen_stream_mutex);

			if (apply_contrast_distribution)
				contrast = contrast_distribution(generator);
			if (apply_brightness_shift_distribution)
				brightness_shift = brightness_shift_distribution(generator);
		}

		unsigned int neuron_count_per_image = original_config.dimension_sizes[0] * original_config.dimension_sizes[1];
		unsigned int image_count = original_config.get_neuron_count() / neuron_count_per_image;
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			cv::Mat1f dest_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), data_transformed + (image_id * neuron_count_per_image));
			cv::Mat1f image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<float *>(data) + (image_id * neuron_count_per_image));

			data_transformer_util::change_brightness_and_contrast(
				dest_image,
				image,
				contrast,
				brightness_shift);
		}
	}
}
