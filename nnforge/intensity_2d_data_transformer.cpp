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
	{
		generator = rnd::get_random_generator();

		contrast_distribution = nnforge_uniform_real_distribution<float>(1.0F / max_contrast_factor, max_contrast_factor);
		brightness_shift_distribution = nnforge_uniform_real_distribution<float>(-max_absolute_brightness_shift, max_absolute_brightness_shift);
	}

	intensity_2d_data_transformer::~intensity_2d_data_transformer()
	{
	}

	void intensity_2d_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("intensity_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("intensity_2d_data_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		float contrast = contrast_distribution.min();
		if (contrast_distribution.max() > contrast_distribution.min())
			contrast = contrast_distribution(generator);
		float brightness_shift = brightness_shift_distribution.min() * 255.0F;
		if (brightness_shift_distribution.max() > brightness_shift_distribution.min())
			brightness_shift = brightness_shift_distribution(generator) * 255.0F;

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(data_transformed) + (neuron_count_per_feature_map * feature_map_id));

			data_transformer_util::change_brightness_and_contrast(
				image,
				contrast,
				brightness_shift);
		}
	}

 	bool intensity_2d_data_transformer::is_deterministic() const
	{
		return false;
	}
}
