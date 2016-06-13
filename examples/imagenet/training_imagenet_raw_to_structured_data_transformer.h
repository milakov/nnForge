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

#include <nnforge/raw_to_structured_data_transformer.h>
#include <nnforge/rnd.h>

#include <boost/thread/thread.hpp>

class training_imagenet_raw_to_structured_data_transformer : public nnforge::raw_to_structured_data_transformer
{
public:
	training_imagenet_raw_to_structured_data_transformer(
		float min_relative_target_area,
		float max_relative_target_area,
		unsigned int target_image_width,
		unsigned int target_image_height,
		float max_aspect_ratio_change);

	virtual ~training_imagenet_raw_to_structured_data_transformer();

	virtual void transform(
		unsigned int sample_id,
		const std::vector<unsigned char>& raw_data,
		float * structured_data);

	virtual nnforge::layer_configuration_specific get_configuration() const;

protected:
	unsigned int target_image_width;
	unsigned int target_image_height;

	boost::mutex gen_mutex;
	nnforge::random_generator gen;
	nnforge_uniform_real_distribution<float> dist_relative_target_area;
	nnforge_uniform_real_distribution<float> dist_log_aspect_ratio;
};
