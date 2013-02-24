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

#include "data_scale_params.h"

namespace nnforge
{
	data_scale_params::data_scale_params()
		: feature_map_count(0)
	{
	}

	data_scale_params::data_scale_params(unsigned int feature_map_count)
		: feature_map_count(feature_map_count)
		, addition_list(feature_map_count, 0.0F)
		, multiplication_list(feature_map_count, 1.0F)
	{
	}

	void data_scale_params::write(std::ostream& output_stream) const
	{
		output_stream.write(reinterpret_cast<const char*>(&feature_map_count), sizeof(feature_map_count));

		output_stream.write(reinterpret_cast<const char*>(&(*addition_list.begin())), sizeof(*addition_list.begin()) * feature_map_count);
		output_stream.write(reinterpret_cast<const char*>(&(*multiplication_list.begin())), sizeof(*multiplication_list.begin()) * feature_map_count);
	}

	void data_scale_params::read(std::istream& input_stream)
	{
		input_stream.read(reinterpret_cast<char*>(&feature_map_count), sizeof(feature_map_count));

		addition_list.resize(feature_map_count);
		multiplication_list.resize(feature_map_count);

		input_stream.read(reinterpret_cast<char*>(&(*addition_list.begin())), sizeof(*addition_list.begin()) * feature_map_count);
		input_stream.read(reinterpret_cast<char*>(&(*multiplication_list.begin())), sizeof(*multiplication_list.begin()) * feature_map_count);
	}
}
