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

#include <mutex>

namespace nnforge
{
	class rotate_band_data_transformer : public data_transformer
	{
	public:
		rotate_band_data_transformer(const std::vector<unsigned int>& max_absolute_band_rotations);

		virtual ~rotate_band_data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
	protected:
		random_generator generator;
		std::mutex gen_stream_mutex;

		std::vector<std::uniform_int_distribution<int> > rotate_band_distributions;
	};
}
