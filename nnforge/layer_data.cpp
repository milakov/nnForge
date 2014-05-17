/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "layer_data.h"

#include <algorithm>

namespace nnforge
{
	layer_data::layer_data()
	{
	}

	void layer_data::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int weight_vector_count = static_cast<unsigned int>(size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&weight_vector_count), sizeof(weight_vector_count));

		for(unsigned int i = 0; i < weight_vector_count; ++i)
		{
			unsigned int weight_count = static_cast<unsigned int>(at(i).size());
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));

			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*at(i).begin())), sizeof(float) * weight_count);
		}
	}

	void layer_data::read(std::istream& binary_stream_to_read_from)
	{
		unsigned int weight_vector_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&weight_vector_count), sizeof(weight_vector_count));

		resize(weight_vector_count);
		for(unsigned int i = 0; i < weight_vector_count; ++i)
		{
			unsigned int weight_count;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));

			at(i).resize(weight_count);

			binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*at(i).begin())), sizeof(float) * weight_count);
		}
	}

	bool layer_data::is_empty() const
	{
		for(std::vector<std::vector<float> >::const_iterator it = begin(); it != end(); it++)
		{
			if (it->size() > 0)
				return false;
		}

		return true;
	}

	void layer_data::apply_dropout_layer_config(
		const dropout_layer_config& dropout,
		bool is_direct)
	{
		for(std::map<unsigned int, float>::const_iterator it = dropout.weight_part_to_dropout_direct_multiplier_map.begin(); it != dropout.weight_part_to_dropout_direct_multiplier_map.end(); ++it)
		{
			unsigned int block_id = it->first;
			float actual_mult = is_direct ? it->second : 1.0F / it->second;

			std::vector<float>& weights = at(block_id);
			mult_transform tr(actual_mult);
			std::transform(weights.begin(), weights.end(), weights.begin(), tr);
		}
	}

	void layer_data::fill(float val)
	{
		for(std::vector<std::vector<float> >::iterator it = begin(); it != end(); ++it)
			std::fill(it->begin(), it->end(), val);
	}

	void layer_data::random_fill(
		float min,
		float max,
		random_generator& gen)
	{
		for(std::vector<std::vector<float> >::iterator it = begin(); it != end(); ++it)
		{
			nnforge_uniform_real_distribution<float> nd(min, max);

			for(std::vector<float>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
				*it2 = nd(gen);
		}
	}

	layer_data::mult_transform::mult_transform(float mult)
		: mult(mult)
	{
	}

	float layer_data::mult_transform::operator() (float in)
	{
		return in * mult;
	}
}
