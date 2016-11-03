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

#include "uniform_intensity_data_transformer.h"

#include "neural_network_exception.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	uniform_intensity_data_transformer::uniform_intensity_data_transformer(
		const std::vector<float>& min_shift_list,
		const std::vector<float>& max_shift_list)
	{
		generator = rnd::get_random_generator();

		for(unsigned int i = 0; i < min_shift_list.size(); ++i)
		{
			bool apply = (min_shift_list[i] < max_shift_list[i]);
			apply_shift_distribution_list.push_back(apply);
			shift_distribution_list.push_back(std::uniform_real_distribution<float>(min_shift_list[i], max_shift_list[i] + (apply ? 0.0F : 1.0F)));
		}
	}

	void uniform_intensity_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.feature_map_count != shift_distribution_list.size())
			throw neural_network_exception((boost::format("uniform_intensity_data_transformer was initialized with %1% distributions and data provided has %2% feature maps") % shift_distribution_list.size() % original_config.feature_map_count).str());

		std::vector<float> shift_list(original_config.feature_map_count);
		{
			std::lock_guard<std::mutex> lock(gen_stream_mutex);

			for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
			{
				std::uniform_real_distribution<float>& dist = shift_distribution_list[feature_map_id];
				float shift = dist.min();
				if (apply_shift_distribution_list[feature_map_id])
					shift = dist(generator);
				shift_list[feature_map_id] = shift;
			}
		}

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			float shift = shift_list[feature_map_id];
			const float * src_data = data + feature_map_id * neuron_count_per_feature_map;
			float * dest_data = data_transformed + feature_map_id * neuron_count_per_feature_map;
			for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				dest_data[i] = src_data[i] + shift;
		}
	}
}
