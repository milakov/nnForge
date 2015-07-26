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
#include <map>
#include <string>

namespace nnforge
{
	class layer_configuration_specific
	{
	public:
		layer_configuration_specific();

		layer_configuration_specific(unsigned int feature_map_count);

		layer_configuration_specific(
			unsigned int feature_map_count,
			const std::vector<unsigned int>& dimension_sizes);

		void write(std::ostream& output_stream) const;

		void read(std::istream& input_stream);

		unsigned int get_neuron_count() const;

		unsigned int get_neuron_count_per_feature_map() const;

		unsigned int get_dimension_count() const;

		unsigned int get_pos(const std::vector<unsigned int>& offsets) const;

		std::vector<unsigned int> get_offsets(unsigned int pos) const;

		// The method throws exception in case the two object are not equal
		void check_equality(const layer_configuration_specific& other) const;

		void check_equality(unsigned int neuron_count) const;

		bool operator==(const layer_configuration_specific& other) const;

		bool operator!=(const layer_configuration_specific& other) const;

	public:
		unsigned int feature_map_count;
		std::vector<unsigned int> dimension_sizes;
	};
}
