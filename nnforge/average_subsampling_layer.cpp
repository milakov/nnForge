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

#include "average_subsampling_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {5D6E4BB2-3891-4FF5-99EF-DA7B5A97CA68}
	const boost::uuids::uuid average_subsampling_layer::layer_guid =
		{ 0x5d, 0x6e, 0x4b, 0xb2
		, 0x38, 0x91
		, 0x4f, 0xf5
		, 0x99, 0xef
		, 0xda, 0x7b, 0x5a, 0x97, 0xca, 0x68 };


	average_subsampling_layer::average_subsampling_layer(const std::vector<unsigned int>& subsampling_sizes)
		: subsampling_sizes(subsampling_sizes)
	{
		if (subsampling_sizes.size() == 0)
			throw neural_network_exception("subsampling sizes for average subsampling layer may not be empty");

		for(unsigned int i = 0; i < subsampling_sizes.size(); i++)
		{
			if (subsampling_sizes[i] == 0)
				throw neural_network_exception("window dimension for average subsampling layer may not be zero");
		}
	}

	const boost::uuids::uuid& average_subsampling_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr average_subsampling_layer::clone() const
	{
		return layer_smart_ptr(new average_subsampling_layer(*this));
	}

	layer_configuration average_subsampling_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if ((input_configuration.dimension_count >= 0) && (input_configuration.dimension_count != static_cast<int>(subsampling_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration.dimension_count).str());

		return layer_configuration(input_configuration.feature_map_count, static_cast<int>(subsampling_sizes.size()));
	}

	layer_configuration_specific average_subsampling_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		if (input_configuration_specific.get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration_specific.get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific.feature_map_count);

		for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
		{
			if (input_configuration_specific.dimension_sizes[i] < subsampling_sizes[i])
				throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than subsampling size (%3%)") % input_configuration_specific.dimension_sizes[i] % i % subsampling_sizes[i]).str());

			res.dimension_sizes.push_back(input_configuration_specific.dimension_sizes[i] / subsampling_sizes[i]);
		}

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > average_subsampling_layer::get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const
	{
		if (output_rectangle_borders.size() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % subsampling_sizes.size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res;

		for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
			res.push_back(std::make_pair(output_rectangle_borders[i].first * subsampling_sizes[i], output_rectangle_borders[i].second * subsampling_sizes[i]));

		return res;
	}

	void average_subsampling_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int dimension_count = static_cast<unsigned int>(subsampling_sizes.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&dimension_count), sizeof(dimension_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*subsampling_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	void average_subsampling_layer::read(std::istream& binary_stream_to_read_from)
	{
		unsigned int dimension_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&dimension_count), sizeof(dimension_count));
		subsampling_sizes.resize(dimension_count);
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*subsampling_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	float average_subsampling_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = 1;
		std::for_each(subsampling_sizes.begin(), subsampling_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float average_subsampling_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();

		return static_cast<float>(neuron_count);
	}
}
