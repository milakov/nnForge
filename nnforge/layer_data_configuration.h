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

#pragma once

#include <vector>
#include <ostream>
#include <istream>

namespace nnforge
{
	class layer_data_configuration
	{
	public:
		layer_data_configuration() = default;

		layer_data_configuration(
			unsigned int input_feature_map_count,
			unsigned int output_feature_map_count,
			const std::vector<unsigned int>& dimension_sizes);

	public:
		unsigned int input_feature_map_count;
		unsigned int output_feature_map_count;
		std::vector<unsigned int> dimension_sizes;
	};

	typedef std::vector<layer_data_configuration> layer_data_configuration_list;
}
