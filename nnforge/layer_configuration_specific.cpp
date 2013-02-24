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

#include "layer_configuration_specific.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <boost/lambda/lambda.hpp>

namespace nnforge
{
	layer_configuration_specific::layer_configuration_specific()
		: feature_map_count(0)
	{
	}

	layer_configuration_specific::layer_configuration_specific(unsigned int feature_map_count)
		: feature_map_count(feature_map_count)
	{
	}

	layer_configuration_specific::layer_configuration_specific(
		unsigned int feature_map_count,
		const std::vector<unsigned int>& dimension_sizes)
		: feature_map_count(feature_map_count), dimension_sizes(dimension_sizes)
	{
	}

	void layer_configuration_specific::write(std::ostream& output_stream) const
	{
		output_stream.write(reinterpret_cast<const char*>(&feature_map_count), sizeof(feature_map_count));

		unsigned int dimension_count = static_cast<unsigned int>(dimension_sizes.size());
		output_stream.write(reinterpret_cast<const char*>(&dimension_count), sizeof(dimension_count));

		output_stream.write(reinterpret_cast<const char*>(&(*dimension_sizes.begin())), sizeof(*dimension_sizes.begin()) * dimension_count);
	}

	void layer_configuration_specific::read(std::istream& input_stream)
	{
		input_stream.read(reinterpret_cast<char*>(&feature_map_count), sizeof(feature_map_count));

		unsigned int dimension_count;
		input_stream.read(reinterpret_cast<char*>(&dimension_count), sizeof(dimension_count));
		dimension_sizes.resize(dimension_count);

		input_stream.read(reinterpret_cast<char*>(&(*dimension_sizes.begin())), sizeof(*dimension_sizes.begin()) * dimension_count);
	}

	unsigned int layer_configuration_specific::get_neuron_count() const
	{
		return get_neuron_count_per_feature_map() * feature_map_count;
	}

	unsigned int layer_configuration_specific::get_neuron_count_per_feature_map() const
	{
		unsigned int neuron_count = 1;
		std::for_each(dimension_sizes.begin(), dimension_sizes.end(), neuron_count *= boost::lambda::_1);

		return neuron_count;
	}

	unsigned int layer_configuration_specific::get_dimension_count() const
	{
		return static_cast<unsigned int>(dimension_sizes.size());
	}

	void layer_configuration_specific::check_equality(const layer_configuration_specific& other) const
	{
		if (feature_map_count != other.feature_map_count)
			throw neural_network_exception((boost::format("Feature map count are not equal for layer configurations: %1% and %2%") % feature_map_count % other.feature_map_count).str());

		if (get_dimension_count() != other.get_dimension_count())
			throw neural_network_exception((boost::format("Dimension count are not equal for layer configurations: %1% and %2%") % get_dimension_count() % other.get_dimension_count()).str());

		for(unsigned int i = 0; i < dimension_sizes.size(); ++i)
		{
			if (dimension_sizes[i] != other.dimension_sizes[i])
				throw neural_network_exception((boost::format("Input configuration sizes of dimension (%1%) are nor equal: %2% and %3%") % i % dimension_sizes[i] % other.dimension_sizes[i]).str());
		}
	}

	void layer_configuration_specific::check_equality(unsigned int neuron_count) const
	{
		if (get_neuron_count() != neuron_count)
			throw neural_network_exception((boost::format("Neuron count are not equal for layer configurations: %1% and %2%") % get_neuron_count() % neuron_count).str());
	}

	bool layer_configuration_specific::operator==(const layer_configuration_specific& other) const
	{
		return !(*this != other);
	}

	bool layer_configuration_specific::operator!=(const layer_configuration_specific& other) const
	{
		if (feature_map_count != other.feature_map_count)
			return true;

		if (get_dimension_count() != other.get_dimension_count())
			return true;

		for(unsigned int i = 0; i < dimension_sizes.size(); ++i)
		{
			if (dimension_sizes[i] != other.dimension_sizes[i])
				return true;
		}

		return false;
	}
}
