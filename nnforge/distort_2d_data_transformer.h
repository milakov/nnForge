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
#include <random>

namespace nnforge
{
	class distort_2d_data_transformer : public data_transformer
	{
	public:
		distort_2d_data_transformer(
			float max_absolute_rotation_angle_in_degrees,
			float max_scale_factor, // >=1
			float min_shift_right_x, // in pixels
			float max_shift_right_x, // in pixels
			float min_shift_down_y, // in pixels
			float max_shift_down_y, // in pixels
			bool flip_around_x_axis_allowed,
			bool flip_around_y_axis_allowed,
			float max_stretch_factor, // >=1
			float min_perspective_distance, // std::numeric_limits<float>::max()
			float border_value = 0.5F); 

		virtual ~distort_2d_data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
	protected:
		float border_value;

		random_generator generator;
		std::mutex gen_stream_mutex;

		bool apply_rotate_angle_distribution;
		std::uniform_real_distribution<float> rotate_angle_distribution;

		bool apply_scale_distribution;
		std::uniform_real_distribution<float> scale_distribution;

		bool apply_shift_x_distribution;
		std::uniform_real_distribution<float> shift_x_distribution;

		bool apply_shift_y_distribution;
		std::uniform_real_distribution<float> shift_y_distribution;

		std::uniform_int_distribution<int> flip_around_x_distribution;
		std::uniform_int_distribution<int> flip_around_y_distribution;

		bool apply_stretch_distribution;
		std::uniform_real_distribution<float> stretch_distribution;

		std::uniform_real_distribution<float> stretch_angle_distribution;

		bool apply_perspective_reverse_distance_distribution;
		std::uniform_real_distribution<float> perspective_reverse_distance_distribution;

		std::uniform_real_distribution<float> perspective_angle_distribution;
	};
}
