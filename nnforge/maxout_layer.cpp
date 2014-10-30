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

#include "maxout_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {5DDE387C-D957-4E98-871D-0550F9FD0CE2}
	const boost::uuids::uuid maxout_layer::layer_guid =
		{ 0x5d, 0xde, 0x38, 0x7c
		, 0xd9, 0x57
		, 0x4e, 0x98
		, 0x87, 0x1d
		, 0x05, 0x50, 0xf9, 0xfd, 0xc, 0xe2 };

	maxout_layer::maxout_layer(unsigned int feature_map_subsampling_size)
		: feature_map_subsampling_size(feature_map_subsampling_size)
	{
		if (feature_map_subsampling_size < 2)
			throw neural_network_exception("Feature map subsampling size should be >= 2 for maxout layer");
	}

	const boost::uuids::uuid& maxout_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr maxout_layer::clone() const
	{
		return layer_smart_ptr(new maxout_layer(*this));
	}

	layer_configuration maxout_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if (input_configuration.feature_map_count >= 0)
		{
			if ((input_configuration.feature_map_count % feature_map_subsampling_size) != 0)
				throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not evenly divisible by feature map subsampling count (%2%)") % input_configuration.feature_map_count % feature_map_subsampling_size).str());

			return layer_configuration(input_configuration.feature_map_count / feature_map_subsampling_size, input_configuration.dimension_count);
		}
		else
		{
			return layer_configuration(input_configuration.feature_map_count, input_configuration.dimension_count);
		}
	}

	layer_configuration_specific maxout_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		if ((input_configuration_specific.feature_map_count % feature_map_subsampling_size) != 0)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not evenly divisible by feature map subsampling count (%2%)") % input_configuration_specific.feature_map_count % feature_map_subsampling_size).str());

		return layer_configuration_specific(input_configuration_specific.feature_map_count / feature_map_subsampling_size, input_configuration_specific.dimension_sizes);
	}

	void maxout_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&feature_map_subsampling_size), sizeof(feature_map_subsampling_size));
	}

	void maxout_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&feature_map_subsampling_size), sizeof(feature_map_subsampling_size));
	}

	float maxout_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = feature_map_subsampling_size - 1;

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float maxout_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(0);
	}
}
