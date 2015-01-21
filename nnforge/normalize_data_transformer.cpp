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

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace nnforge
{
	// {9BABFBB9-3502-4C58-AA38-A550C1AA843D}
	const boost::uuids::uuid normalize_data_transformer::normalizer_guid =
	{ 0x9b, 0xab, 0xfb, 0xb9
	, 0x35, 0x02
	, 0x4c, 0x58
	, 0xaa, 0x38
	, 0xa5, 0x50, 0xc1, 0xaa, 0x84, 0x3d };

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
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_float)
			throw neural_network_exception("normalize_data_transformer is implemented for data stored as floats only");

		float * dt = static_cast<float *>(data_transformed);
		unsigned int elem_count_per_feature_map = original_config.get_neuron_count_per_feature_map();

		for(std::vector<std::pair<float, float> >::const_iterator mul_add_it = mul_add_list.begin(); mul_add_it != mul_add_list.end(); ++mul_add_it, dt += elem_count_per_feature_map)
			std::transform(dt, dt + elem_count_per_feature_map, dt, normalize_helper_struct(mul_add_it->first, mul_add_it->second));
	}

	void normalize_data_transformer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		binary_stream_to_write_to.write(reinterpret_cast<const char*>(normalizer_guid.data), sizeof(normalizer_guid.data));

		unsigned int feature_map_count = static_cast<unsigned int>(mul_add_list.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&feature_map_count), sizeof(feature_map_count));

		for(std::vector<std::pair<float, float> >::const_iterator it =  mul_add_list.begin(); it != mul_add_list.end(); ++it)
		{
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(it->first)), sizeof(float));
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(it->second)), sizeof(float));
		}

		binary_stream_to_write_to.flush();
	}

	void normalize_data_transformer::read(std::istream& binary_stream_to_read_from)
	{
 		mul_add_list.clear();

		binary_stream_to_read_from.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid normalizer_guid_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(normalizer_guid_read.data), sizeof(normalizer_guid_read.data));
		if (normalizer_guid_read != normalizer_guid)
			throw neural_network_exception((boost::format("Unknown normalizer GUID encountered in input stream: %1%") % normalizer_guid_read).str());

		unsigned int feature_map_count_read;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&feature_map_count_read), sizeof(feature_map_count_read));

		for(unsigned int i = 0; i < feature_map_count_read; ++i)
		{
			float mult;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&mult), sizeof(float));
			float add;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&add), sizeof(float));

			mul_add_list.push_back(std::make_pair(mult, add));
		}
	}

 	bool normalize_data_transformer::is_deterministic() const
	{
		return true;
	}
}
