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

#include "convolution_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
 #include "nn_types.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {5957B44F-699E-4DDB-836E-3FB3EEB54965}
	const boost::uuids::uuid convolution_layer::layer_guid =
		{ 0x59, 0x57, 0xb4, 0x4f
		, 0x69, 0x9e
		, 0x4d, 0xdb
		, 0x83, 0x6e
		, 0x3f, 0xb3, 0xee, 0xb5, 0x49, 0x65 };

	// {8AD07635-DDFE-43B4-B26A-15CA19155A65}
	const boost::uuids::uuid convolution_layer::layer_guid_v1 =
		{ 0x8a, 0xd0, 0x76, 0x35
		, 0xdd, 0xfe
		, 0x43, 0xb4
		, 0xb2, 0x6a
		, 0x15, 0xca, 0x19, 0x15, 0x5a, 0x65 };

	convolution_layer::convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count,
		const std::vector<unsigned int>& left_zero_padding,
		const std::vector<unsigned int>& right_zero_padding)
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

		if ((left_zero_padding.size() != 0) && (left_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for left zero padding") % left_zero_padding.size()).str());
		if ((right_zero_padding.size() != 0) && (right_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for right zero padding") % right_zero_padding.size()).str());

		if (left_zero_padding.empty())
			this->left_zero_padding.resize(window_sizes.size(), 0);
		else
		{
			for(unsigned int i = 0; i < window_sizes.size(); i++)
				if (left_zero_padding[i] >= window_sizes[i])
					throw neural_network_exception((boost::format("left zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % left_zero_padding[i] % i % window_sizes[i]).str());
			this->left_zero_padding = left_zero_padding;
		}

		if (right_zero_padding.empty())
			this->right_zero_padding.resize(window_sizes.size(), 0);
		else
		{
			for(unsigned int i = 0; i < window_sizes.size(); i++)
				if (right_zero_padding[i] >= window_sizes[i])
					throw neural_network_exception((boost::format("right zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % right_zero_padding[i] % i % window_sizes[i]).str());
			this->right_zero_padding = right_zero_padding;
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
			unsigned int total_input_dimension_size = input_configuration_specific.dimension_sizes[i] + left_zero_padding[i] + right_zero_padding[i];
			if (total_input_dimension_size < window_sizes[i])
				throw neural_network_exception((boost::format("Too small total dimension size (with padding) %1% of dimension (%2%) is smaller than layer window size (%3%)") % total_input_dimension_size % i % window_sizes[i]).str());

			res.dimension_sizes.push_back(total_input_dimension_size + 1 - window_sizes[i]);
		}

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > convolution_layer::get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const
	{
		if (output_rectangle_borders.size() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % window_sizes.size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res;

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
			res.push_back(
				std::make_pair(
					static_cast<unsigned int>(std::max(0, static_cast<int>(output_rectangle_borders[i].first) - static_cast<int>(left_zero_padding[i]))),
					(output_rectangle_borders[i].second + window_sizes[i] - 1) - left_zero_padding[i]
				)
			);

		return res;
	}

	void convolution_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&output_feature_map_count), sizeof(output_feature_map_count));

		unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&dimension_count), sizeof(dimension_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*left_zero_padding.begin())), sizeof(unsigned int) * dimension_count);
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*right_zero_padding.begin())), sizeof(unsigned int) * dimension_count);
	}

	void convolution_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&output_feature_map_count), sizeof(output_feature_map_count));

		unsigned int dimension_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&dimension_count), sizeof(dimension_count));
		window_sizes.resize(dimension_count);
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);

		left_zero_padding.resize(dimension_count, 0);
		right_zero_padding.resize(dimension_count, 0);
		if (layer_read_guid != layer_guid_v1)
		{
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*left_zero_padding.begin())), sizeof(unsigned int) * dimension_count);
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*right_zero_padding.begin())), sizeof(unsigned int) * dimension_count);
		}
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
		layer_data_custom& data_custom,
		random_generator& generator) const
	{
		unsigned int weight_count = 1;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);

		float average_feature_map_count = sqrtf(static_cast<float>(input_feature_map_count) * static_cast<float>(output_feature_map_count));

		float standard_deviation = sqrtf(1.0F / (average_feature_map_count * static_cast<float>(weight_count)));
		float max_abs_value = 100.0F * standard_deviation;

		nnforge_normal_distribution<float> nd(0.0F, standard_deviation);

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

	float convolution_layer::get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	layer_data_configuration_list convolution_layer::get_layer_data_configuration_list() const
	{
		layer_data_configuration_list res;
		res.push_back(layer_data_configuration(input_feature_map_count, output_feature_map_count, window_sizes));
		res.push_back(layer_data_configuration(1, output_feature_map_count, std::vector<unsigned int>()));

		return res;
	}

	std::set<unsigned int> convolution_layer::get_weight_decay_part_id_set() const
	{
		std::set<unsigned int> res;
		res.insert(0);
		return res;
	}
}
