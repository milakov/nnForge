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
			shift_distribution_list.push_back(nnforge_uniform_real_distribution<float>(min_shift_list[i], max_shift_list[i]));
	}

	uniform_intensity_data_transformer::~uniform_intensity_data_transformer()
	{
	}

	void uniform_intensity_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_float)
			throw neural_network_exception("uniform_intensity_data_transformer is implemented for data stored as floats only");

		if (original_config.feature_map_count != shift_distribution_list.size())
			throw neural_network_exception((boost::format("uniform_intensity_data_transformer was initialized with %1% distributions and data provided has %2% feature maps") % shift_distribution_list.size() % original_config.feature_map_count).str());

		float * data_typed = static_cast<float *>(data_transformed);

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			nnforge_uniform_real_distribution<float>& dist = shift_distribution_list[feature_map_id];
			float shift = dist.min();
			if (dist.max() > dist.min())
				shift = dist(generator);

			float * dest_data = data_typed + feature_map_id * neuron_count_per_feature_map;
			for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				dest_data[i] += shift;
		}
	}

 	bool uniform_intensity_data_transformer::is_deterministic() const
	{
		return false;
	}
}
