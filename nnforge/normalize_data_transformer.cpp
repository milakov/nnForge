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

#include "normalize_data_transformer.h"

#include "neural_network_exception.h"
#include "proto/data_normalizer.pb.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace nnforge
{
	normalize_data_transformer::normalize_data_transformer()
	{
	}

	normalize_data_transformer::normalize_data_transformer(const std::vector<feature_map_data_stat>& feature_map_data_stat_list)
	{
		for (std::vector<feature_map_data_stat>::const_iterator it = feature_map_data_stat_list.begin(); it != feature_map_data_stat_list.end(); ++it)
		{
			const feature_map_data_stat& current_stat = *it;
			float mult;
			float add;
			if (current_stat.std_dev > 0.0F)
			{
				mult = 1.0F / current_stat.std_dev;
				add = -current_stat.average * mult;
			}
			else
			{
				mult = 1.0F;
				add = 0.0F;
			}

			mul_add_list.push_back(std::make_pair(mult, add));
		}
	}

	normalize_data_transformer::~normalize_data_transformer()
	{
	}

	nnforge_shared_ptr<normalize_data_transformer> normalize_data_transformer::get_inverted_transformer() const
	{
		nnforge_shared_ptr<normalize_data_transformer> res(new normalize_data_transformer());

		for(std::vector<std::pair<float, float> >::const_iterator mul_add_it = mul_add_list.begin(); mul_add_it != mul_add_list.end(); ++mul_add_it)
		{
			float new_mult = 1.0F / mul_add_it->first;
			float new_add = -new_mult * mul_add_it->second;

			res->mul_add_list.push_back(std::make_pair(new_mult, new_add));
		}

		return res;
	}

	void normalize_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		unsigned int elem_count_per_feature_map = original_config.get_neuron_count_per_feature_map();

		for(std::vector<std::pair<float, float> >::const_iterator mul_add_it = mul_add_list.begin(); mul_add_it != mul_add_list.end(); ++mul_add_it, data += elem_count_per_feature_map, data_transformed += elem_count_per_feature_map)
			std::transform(data, data + elem_count_per_feature_map, data_transformed, normalize_helper_struct(mul_add_it->first, mul_add_it->second));
	}

	void normalize_data_transformer::write_proto(std::ostream& stream_to_write_to) const
	{
		protobuf::DataNormalizer normalizer;

		for(std::vector<std::pair<float, float> >::const_iterator it =  mul_add_list.begin(); it != mul_add_list.end(); ++it)
		{
			protobuf::DataNormalizer_FeatureMapParam * feature_map_param = normalizer.add_feature_map_param();
			feature_map_param->set_mul(it->first);
			feature_map_param->set_add(it->second);
		}

		google::protobuf::io::OstreamOutputStream output_stream(&stream_to_write_to);
		google::protobuf::TextFormat::Print(normalizer, &output_stream);
	}

	void normalize_data_transformer::read_proto(std::istream& stream_to_read_from)
	{
 		mul_add_list.clear();

		protobuf::DataNormalizer normalizer;
		google::protobuf::io::IstreamInputStream input_stream(&stream_to_read_from);
		google::protobuf::TextFormat::Parse(&input_stream, &normalizer);

		for(int i = 0; i < normalizer.feature_map_param_size(); ++i)
		{
			mul_add_list.push_back(std::make_pair(normalizer.feature_map_param(i).mul(), normalizer.feature_map_param(i).add()));
		}
	}
}
