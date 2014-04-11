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

#include "convolution_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
 #include "nn_types.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {8AD07635-DDFE-43B4-B26A-15CA19155A65}
	const boost::uuids::uuid convolution_layer::layer_guid =
		{ 0x8a, 0xd0, 0x76, 0x35
		, 0xdd, 0xfe
		, 0x43, 0xb4
		, 0xb2, 0x6a
		, 0x15, 0xca, 0x19, 0x15, 0x5a, 0x65 };

	convolution_layer::convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count)
		: window_sizes(window_sizes),
		input_feature_map_count(input_feature_map_count),
		output_feature_map_count(output_feature_map_count)
	{
		if (window_sizes.size() == 0)
			throw neural_network_exception("window sizes for convolution layer may not be empty");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
		{
			if (window_sizes[i] == 0)
				throw neural_network_exception("window dimension for convolution layer may not be zero");
		}
	}

	const boost::uuids::uuid& convolution_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr convolution_layer::clone() const
	{
		return layer_smart_ptr(new convolution_layer(*this));
	}

	layer_configuration convolution_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if ((input_configuration.feature_map_count >= 0) && (input_configuration.feature_map_count != static_cast<int>(input_feature_map_count)))
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration.feature_map_count).str());

		if ((input_configuration.dimension_count >= 0) && (input_configuration.dimension_count != static_cast<int>(window_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration.dimension_count).str());

		return layer_configuration(output_feature_map_count, static_cast<int>(window_sizes.size()));
	}

	layer_configuration_specific convolution_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		if (input_configuration_specific.feature_map_count != input_feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration_specific.feature_map_count).str());

		if (input_configuration_specific.get_dimension_count() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration_specific.get_dimension_count()).str());

		layer_configuration_specific res(output_feature_map_count);

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
		{
			if (input_configuration_specific.dimension_sizes[i] < window_sizes[i])
				throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than layer window size (%3%)") % input_configuration_specific.dimension_sizes[i] % i % window_sizes[i]).str());

			res.dimension_sizes.push_back(input_configuration_specific.dimension_sizes[i] + 1 - window_sizes[i]);
		}

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > convolution_layer::get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const
	{
		if (output_rectangle_borders.size() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % window_sizes.size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res;

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
			res.push_back(std::make_pair(output_rectangle_borders[i].first, output_rectangle_borders[i].second + window_sizes[i] - 1));

		return res;
	}

	void convolution_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&output_feature_map_count), sizeof(output_feature_map_count));

		unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&dimension_count), sizeof(dimension_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	void convolution_layer::read(std::istream& binary_stream_to_read_from)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&output_feature_map_count), sizeof(output_feature_map_count));

		unsigned int dimension_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&dimension_count), sizeof(dimension_count));
		window_sizes.resize(dimension_count);
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	data_config convolution_layer::get_data_config() const
	{
		data_config res;

		unsigned int weight_count = input_feature_map_count * output_feature_map_count;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);

		res.push_back(weight_count);

		res.push_back(output_feature_map_count);

		return res;
	}

	void convolution_layer::randomize_data(
		layer_data& data,
		random_generator& generator) const
	{
		unsigned int input_neuron_count = input_feature_map_count;
		std::for_each(window_sizes.begin(), window_sizes.end(), input_neuron_count *= boost::lambda::_1);

		float standard_deviation = 1.0F / sqrtf(static_cast<float>(input_neuron_count));
		float max_abs_value = 3.0F * standard_deviation;

		nnforge_normal_distribution<float> nd(0.0F, standard_deviation);
		//nnforge_uniform_real_distribution<float> nd(-2.0F * standard_deviation, 2.0F * standard_deviation);

		for(unsigned int i = 0; i < data[0].size(); ++i)
		{
			float val = nd(generator);
			while (fabs(val) > max_abs_value)
				val = nd(generator);

			data[0][i] = val;
		}

		std::fill(data[1].begin(), data[1].end(), 0.0F);
	}

	float convolution_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);
		per_item_flops -= 1;

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_weights_update_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return (static_cast<float>(neuron_count) * static_cast<float>(per_item_flops)) + static_cast<float>(input_configuration_specific.get_neuron_count());
	}

	dropout_layer_config convolution_layer::get_dropout_layer_config(float dropout_rate) const
	{
		if ((dropout_rate < 0.0F) || (dropout_rate >= 1.0F))
			throw neural_network_exception((boost::format("Illegal dropout rate: %1%") % dropout_rate).str());

		dropout_layer_config res;
		res.weight_part_to_dropout_direct_multiplier_map.insert(std::make_pair<unsigned int, float>(0, 1.0F - dropout_rate));

		return res;
	}

	layer_data_configuration_list convolution_layer::get_layer_data_configuration_list() const
	{
		layer_data_configuration_list res;
		res.push_back(layer_data_configuration(input_feature_map_count, output_feature_map_count, window_sizes));
		res.push_back(layer_data_configuration(1, output_feature_map_count, std::vector<unsigned int>()));

		return res;
	}
}
