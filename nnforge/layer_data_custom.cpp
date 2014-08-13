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

#include "layer_data_custom.h"

#include <algorithm>

namespace nnforge
{
	layer_data_custom::layer_data_custom()
	{
	}

	void layer_data_custom::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int weight_vector_count = static_cast<unsigned int>(size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&weight_vector_count), sizeof(weight_vector_count));

		for(unsigned int i = 0; i < weight_vector_count; ++i)
		{
			unsigned int weight_count = static_cast<unsigned int>(at(i).size());
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));

			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*at(i).begin())), sizeof(int) * weight_count);
		}
	}

	void layer_data_custom::read(std::istream& binary_stream_to_read_from)
	{
		unsigned int weight_vector_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&weight_vector_count), sizeof(weight_vector_count));

		resize(weight_vector_count);
		for(unsigned int i = 0; i < weight_vector_count; ++i)
		{
			unsigned int weight_count;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));

			at(i).resize(weight_count);

			binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*at(i).begin())), sizeof(int) * weight_count);
		}
	}
}
