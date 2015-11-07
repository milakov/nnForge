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

#include "data_transformer.h"
#include "rnd.h"
#include "nn_types.h"

#include <boost/thread/thread.hpp>

namespace nnforge
{
	class intensity_2d_data_transformer : public data_transformer
	{
	public:
		intensity_2d_data_transformer(
			float max_contrast_factor, // >=1
			float max_absolute_brightness_shift); // in [0,1]

		virtual ~intensity_2d_data_transformer();

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
	protected:
		random_generator generator;
		boost::mutex gen_stream_mutex;

		nnforge_uniform_real_distribution<float> contrast_distribution;
		nnforge_uniform_real_distribution<float> brightness_shift_distribution;
	};
}
