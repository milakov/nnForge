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

#include "rotate_band_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

#include <memory>

namespace nnforge
{
	rotate_band_data_transformer::rotate_band_data_transformer(const std::vector<unsigned int>& max_absolute_band_rotations)
	{
		generator = rnd::get_random_generator();

		for(std::vector<unsigned int>::const_iterator it = max_absolute_band_rotations.begin(); it != max_absolute_band_rotations.end(); ++it)
			rotate_band_distributions.push_back(nnforge_uniform_int_distribution<int>(-static_cast<int>(*it), static_cast<int>(*it)));
	}

	rotate_band_data_transformer::~rotate_band_data_transformer()
	{
	}

	void rotate_band_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		const std::vector<unsigned int>& dimension_sizes = original_config.dimension_sizes;

		if (dimension_sizes.size() != rotate_band_distributions.size())
			throw neural_network_exception((boost::format("rotate_band_data_transformer is created with %1%-dimensional rotations, data has %2% dimensions") % rotate_band_distributions.size() % dimension_sizes.size()).str());

		size_t elem_size = neuron_data_type::get_input_size(type);

		const unsigned char * src_begin = (const unsigned char *)data;
		unsigned char * dst = (unsigned char *)data_transformed;

		std::vector<unsigned int> src_pos_list;
		std::vector<unsigned int>::const_iterator it2 = dimension_sizes.begin();
		for(std::vector<nnforge_uniform_int_distribution<int> >::iterator it = rotate_band_distributions.begin(); it != rotate_band_distributions.end(); ++it, ++it2)
		{
			nnforge_uniform_int_distribution<int>& rotate_band_distribution = *it;
			int rotate_band = rotate_band_distribution.min();
			if (rotate_band_distribution.max() > rotate_band_distribution.min())
				rotate_band = rotate_band_distribution(generator);
			if (rotate_band < 0)
				rotate_band += *it2;
			src_pos_list.push_back(rotate_band);
		}

		std::vector<unsigned int> dst_pos_list(dimension_sizes.size(), 0);

		unsigned int x_xopy_count1 = dimension_sizes[0] - src_pos_list[0];
		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			while (true)
			{
				unsigned int offset = 0;
				for(int i = dimension_sizes.size() - 1; i > 0; --i)
					offset = (offset + src_pos_list[i]) * dimension_sizes[i - 1];
				const unsigned char * src = src_begin + offset * elem_size;

				memcpy(dst, src + src_pos_list[0] * elem_size, x_xopy_count1 * elem_size);
				if (src_pos_list[0] > 0)
					memcpy(dst + x_xopy_count1 * elem_size, src, src_pos_list[0] * elem_size);
				dst += dimension_sizes[0] * elem_size;

				bool inc = false;
				for(int i = 1; i < dimension_sizes.size(); ++i)
				{
					dst_pos_list[i]++;
					src_pos_list[i]++;
					if (src_pos_list[i] == dimension_sizes[i])
						src_pos_list[i] = 0;
					if (dst_pos_list[i] < dimension_sizes[i])
					{
						inc = true;
						break;
					}
					else
						dst_pos_list[i] = 0;
				}
				if (!inc)
					break;
			}

			src_begin += original_config.get_neuron_count_per_feature_map() * elem_size;
		}
	}

	bool rotate_band_data_transformer::is_in_place() const
	{
		return false;
	}
}
