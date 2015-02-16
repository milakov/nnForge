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

#include "parametric_rectified_linear_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	// {B31C91D1-ED44-47F7-A68E-91513F888544}
	const boost::uuids::uuid parametric_rectified_linear_layer::layer_guid =
		{ 0xb3, 0x1c, 0x91, 0xd1
		, 0xed, 0x44
		, 0x47, 0xf7
		, 0xa6, 0x8e
		, 0x91, 0x51, 0x3f, 0x88, 0x85, 0x44 };

	parametric_rectified_linear_layer::parametric_rectified_linear_layer(unsigned int feature_map_count)
		: feature_map_count(feature_map_count)
	{
	}

	const boost::uuids::uuid& parametric_rectified_linear_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr parametric_rectified_linear_layer::clone() const
	{
		return layer_smart_ptr(new parametric_rectified_linear_layer(*this));
	}

	float parametric_rectified_linear_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 2);
	}

	float parametric_rectified_linear_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 2);
	}

	float parametric_rectified_linear_layer::get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 3);
	}

	layer_configuration parametric_rectified_linear_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if (input_configuration.feature_map_count >= 0)
		{
			if (input_configuration.feature_map_count != feature_map_count)
				throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not rqual to feature map count (%2%) in perametric_rectified_linear_layer") % input_configuration.feature_map_count % feature_map_count).str());
		}

		return input_configuration;
	}

	layer_configuration_specific parametric_rectified_linear_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		if (input_configuration_specific.feature_map_count != feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not rqual to feature map count (%2%) in perametric_rectified_linear_layer") % input_configuration_specific.feature_map_count % feature_map_count).str());

		return input_configuration_specific;
	}

	void parametric_rectified_linear_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&feature_map_count), sizeof(feature_map_count));
	}

	void parametric_rectified_linear_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&feature_map_count), sizeof(feature_map_count));
	}

	data_config parametric_rectified_linear_layer::get_data_config() const
	{
		data_config res;

		res.push_back(feature_map_count);

		return res;
	}

	void parametric_rectified_linear_layer::randomize_data(
		layer_data& data,
		layer_data_custom& data_custom,
		random_generator& generator) const
	{
		std::fill(data[0].begin(), data[0].end(), 0.25F);
	}

	layer_data_configuration_list parametric_rectified_linear_layer::get_layer_data_configuration_list() const
	{
		layer_data_configuration_list res;

		res.push_back(layer_data_configuration(feature_map_count, 1, std::vector<unsigned int>()));

		return res;
	}

	std::set<unsigned int> parametric_rectified_linear_layer::get_weight_decay_part_id_set() const
	{
		return std::set<unsigned int>();
	}
}
