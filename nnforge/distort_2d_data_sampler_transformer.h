/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include <memory>

namespace nnforge
{
	struct distort_2d_data_sampler_param
	{
		float rotation_angle_in_degrees;
		float scale;
		float shift_right_x;
		float shift_down_y;
	};

	class distort_2d_data_sampler_transformer : public data_transformer
	{
	public:
		distort_2d_data_sampler_transformer(const std::vector<distort_2d_data_sampler_param>& params);

		distort_2d_data_sampler_transformer(
			const std::vector<float>& rotation_angle_in_degrees_list,
			const std::vector<float>& scale_list, // 1.0F - no scaling
			const std::vector<float>& shift_right_x_list,
			const std::vector<float>& shift_down_y_list);

		virtual ~distort_2d_data_sampler_transformer();

		virtual void transform(
			const void * data,
			void * data_transformed,
			neuron_data_type::input_type type,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
		virtual bool is_in_place() const;

		virtual unsigned int get_sample_count() const;

		virtual bool is_deterministic() const;

	protected:
		std::vector<distort_2d_data_sampler_param> params;
	};
}
