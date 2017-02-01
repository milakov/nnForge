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

#include <opencv2/core/core.hpp>
#include <mutex>
#include <random>

namespace nnforge
{
	class elastic_deformation_2d_data_transformer : public data_transformer
	{
	public:
		elastic_deformation_2d_data_transformer(
			float sigma, // >0
			float alpha,
			float border_value = 0.5F);

		virtual ~elastic_deformation_2d_data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
		static void smooth(
			cv::Mat1f disp,
			int ksize,
			float sigma,
			float alpha,
			bool is_x,
			float disp_base_pos = 0.0F,
			float disp_step = 1.0F);

	protected:
		float alpha;
		float sigma;
		float border_value;

		random_generator gen;
		std::mutex gen_stream_mutex;
		std::uniform_real_distribution<float> displacement_distribution;
	};
}
