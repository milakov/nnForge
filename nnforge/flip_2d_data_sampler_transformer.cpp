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

#include "flip_2d_data_sampler_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>
#include <cstring>

namespace nnforge
{
	flip_2d_data_sampler_transformer::flip_2d_data_sampler_transformer(unsigned int flip_around_dimension_id)
		: flip_around_dimension_id(flip_around_dimension_id)
	{
	}

	flip_2d_data_sampler_transformer::~flip_2d_data_sampler_transformer()
	{
	}

	void flip_2d_data_sampler_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("flip_2d_data_sampler_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("flip_2d_data_sampler_transformer is processing 2d data only, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		unsigned int neuron_count_per_feature_map = original_config.get_neuron_count_per_feature_map();
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			cv::Mat1b src_image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<unsigned char *>(static_cast<const unsigned char *>(data)) + (neuron_count_per_feature_map * feature_map_id));
			cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(data_transformed) + (neuron_count_per_feature_map * feature_map_id));
			memcpy(
				((unsigned char *)data_transformed) + neuron_count_per_feature_map * feature_map_id,
				((unsigned char *)data) + neuron_count_per_feature_map * feature_map_id,
				neuron_count_per_feature_map * neuron_data_type::get_input_size(type));

			if (sample_id == 1)
			{
				data_transformer_util::flip(
					image,
					(flip_around_dimension_id == 0),
					(flip_around_dimension_id == 1));
			}
		}
	}

	bool flip_2d_data_sampler_transformer::is_in_place() const
	{
		return false;
	}

	unsigned int flip_2d_data_sampler_transformer::get_sample_count() const
	{
		return 2;
	}
}
