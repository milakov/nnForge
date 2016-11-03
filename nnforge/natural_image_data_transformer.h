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

#pragma once

#include "data_transformer.h"
#include "rnd.h"

#include <vector>
#include <mutex>

namespace nnforge
{
	class natural_image_data_transformer : public data_transformer
	{
	public:
		natural_image_data_transformer(
			float brightness = 0.4F,
			float contrast = 0.4F,
			float saturation = 0.4F,
			float lighting = 0.1F);

		virtual ~natural_image_data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
	private:
		enum augmentation_type
		{
			augmentation_brightness,
			augmentation_contrast,
			augmentation_saturation
		};

	protected:
		random_generator generator;
		std::mutex gen_mutex;

		bool apply_brightness_distribution;
		std::uniform_real_distribution<float> brightness_distribution;

		bool apply_contrast_distribution;
		std::uniform_real_distribution<float> contrast_distribution;

		bool apply_saturation_distribution;
		std::uniform_real_distribution<float> saturation_distribution;

		bool apply_lighting;
		std::normal_distribution<float> lighting_1st_eigen_alpha_distribution;
		std::normal_distribution<float> lighting_2nd_eigen_alpha_distribution;
		std::normal_distribution<float> lighting_3rd_eigen_alpha_distribution;
	};
}
