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

#include "embed_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	embed_data_transformer::embed_data_transformer(
		const std::vector<unsigned int>& output_sizes,
		const std::vector<unsigned int>& left_padding,
		unsigned char padding_value)
		: output_sizes(output_sizes)
		, left_padding(left_padding)
		, padding_value(padding_value)
	{
		if (output_sizes.size() != left_padding.size())
			throw neural_network_exception((boost::format("embed_data_transformer is created with different output_sizes and padding dimensions: %1% and %2%") % output_sizes.size() % left_padding.size()).str());
	}

	embed_data_transformer::~embed_data_transformer()
	{
	}

	void embed_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("embed_data_transformer is implemented for data stored as bytes only");

		const std::vector<unsigned int>& dimension_sizes = original_config.dimension_sizes;

		if (dimension_sizes.size() != output_sizes.size())
			throw neural_network_exception((boost::format("embed_data_transformer is created with %1%-dimensions, data has %2% dimensions") % output_sizes.size() % dimension_sizes.size()).str());

		std::vector<unsigned int> dst_offset_list;
		for(unsigned int i = 0; i < dimension_sizes.size(); ++i)
		{
			if (dimension_sizes[i] < output_sizes[i] + left_padding[i])
				throw neural_network_exception((boost::format("Dimension %1% of original config has %2% size while minimum is %3%") % i % dimension_sizes[i] % (output_sizes[i] - left_padding[i])).str());
			dst_offset_list.push_back(left_padding[i]);
		}

		std::vector<unsigned int> src_pos_list(dimension_sizes.size(), 0);

		const unsigned char * src = (const unsigned char *)data;
		unsigned char * dst_begin = (unsigned char *)data_transformed;

		layer_configuration_specific output_config = get_transformed_configuration(original_config);

		memset(data_transformed, padding_value, output_config.get_neuron_count());

		for(unsigned int feature_map_id = 0; feature_map_id < original_config.feature_map_count; ++feature_map_id)
		{
			while (true)
			{
				unsigned int dst_offset = src_pos_list.back() + dst_offset_list.back();
				for(int i = static_cast<int>(dimension_sizes.size()) - 2; i >= 0; --i)
					dst_offset = dst_offset * dimension_sizes[i] + src_pos_list[i] + dst_offset_list[i];

				memcpy(dst_begin + dst_offset, src, dimension_sizes[0]);
				src += dimension_sizes[0];

				bool inc = false;
				for(int i = 1; i < dimension_sizes.size(); ++i)
				{
					src_pos_list[i]++;
					if (src_pos_list[i] < dimension_sizes[i])
					{
						inc = true;
						break;
					}
					else
						src_pos_list[i] = 0;
				}
				if (!inc)
					break;
			}

			dst_begin += output_config.get_neuron_count_per_feature_map();
		}
	}

	layer_configuration_specific embed_data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		layer_configuration_specific res(original_config.feature_map_count);
		res.dimension_sizes = output_sizes;

		return res;
	}

	bool embed_data_transformer::is_in_place() const
	{
		return false;
	}

 	bool embed_data_transformer::is_deterministic() const
	{
		return true;
	}
}
