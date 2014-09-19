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

#include "sparse_convolution_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
 #include "nn_types.h"

#include <algorithm>
#include <set>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {359B361C-61E7-4E52-89E6-E722B433F95C}
	const boost::uuids::uuid sparse_convolution_layer::layer_guid =
		{ 0x35, 0x9b, 0x36, 0x1c
		, 0x61, 0xe7
		, 0x4e, 0x52
		, 0x89, 0xe6
		, 0xe7, 0x22, 0xb4, 0x33, 0xf9, 0x5c };

	sparse_convolution_layer::sparse_convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count,
		unsigned int feature_map_connection_count)
		: window_sizes(window_sizes),
		input_feature_map_count(input_feature_map_count),
		output_feature_map_count(output_feature_map_count),
		feature_map_connection_count(feature_map_connection_count)
	{
		check_consistency();
	}

	sparse_convolution_layer::sparse_convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count,
		float feature_map_connection_sparsity_ratio)
		: window_sizes(window_sizes),
		input_feature_map_count(input_feature_map_count),
		output_feature_map_count(output_feature_map_count),
		feature_map_connection_count(static_cast<unsigned int>(input_feature_map_count * output_feature_map_count * feature_map_connection_sparsity_ratio))
	{
		check_consistency();
	}

	void sparse_convolution_layer::check_consistency()
	{
		if (window_sizes.size() == 0)
			throw neural_network_exception("window sizes for sparse convolution layer may not be empty");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
		{
			if (window_sizes[i] == 0)
				throw neural_network_exception("window dimension for sparse convolution layer may not be zero");
		}

		if (feature_map_connection_count < input_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be smaller than input_feature_map_count");
		if (feature_map_connection_count < output_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be smaller than output_feature_map_count");
		if (feature_map_connection_count > input_feature_map_count * output_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be larger than in dense case");
	}

	const boost::uuids::uuid& sparse_convolution_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr sparse_convolution_layer::clone() const
	{
		return layer_smart_ptr(new sparse_convolution_layer(*this));
	}

	layer_configuration sparse_convolution_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if ((input_configuration.feature_map_count >= 0) && (input_configuration.feature_map_count != static_cast<int>(input_feature_map_count)))
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration.feature_map_count).str());

		if ((input_configuration.dimension_count >= 0) && (input_configuration.dimension_count != static_cast<int>(window_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration.dimension_count).str());

		return layer_configuration(output_feature_map_count, static_cast<int>(window_sizes.size()));
	}

	layer_configuration_specific sparse_convolution_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
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

	std::vector<std::pair<unsigned int, unsigned int> > sparse_convolution_layer::get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const
	{
		if (output_rectangle_borders.size() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % window_sizes.size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res;

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
			res.push_back(std::make_pair(output_rectangle_borders[i].first, output_rectangle_borders[i].second + window_sizes[i] - 1));

		return res;
	}

	void sparse_convolution_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&output_feature_map_count), sizeof(output_feature_map_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&feature_map_connection_count), sizeof(feature_map_connection_count));

		unsigned int dimension_count = static_cast<unsigned int>(window_sizes.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&dimension_count), sizeof(dimension_count));
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	void sparse_convolution_layer::read(std::istream& binary_stream_to_read_from)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&input_feature_map_count), sizeof(input_feature_map_count));
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&output_feature_map_count), sizeof(output_feature_map_count));
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&feature_map_connection_count), sizeof(feature_map_connection_count));

		unsigned int dimension_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&dimension_count), sizeof(dimension_count));
		window_sizes.resize(dimension_count);
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&(*window_sizes.begin())), sizeof(unsigned int) * dimension_count);
	}

	data_config sparse_convolution_layer::get_data_config() const
	{
		data_config res;

		unsigned int weight_count = feature_map_connection_count;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);

		res.push_back(weight_count);
		res.push_back(output_feature_map_count);

		return res;
	}

	data_custom_config sparse_convolution_layer::get_data_custom_config() const
	{
		data_custom_config res;

		res.push_back(feature_map_connection_count); // column indices
		res.push_back(output_feature_map_count + 1); // row indices

		return res;
	}

	void sparse_convolution_layer::randomize_data(
		layer_data& data,
		layer_data_custom& data_custom,
		random_generator& generator) const
	{
		randomize_custom_data(data_custom, generator);

		randomize_weights(data, data_custom, generator);
	}

	void sparse_convolution_layer::randomize_weights(
		layer_data& data,
		const layer_data_custom& data_custom,
		random_generator& generator) const
	{
		unsigned int input_neuron_count_per_feature_map = 1;
		std::for_each(window_sizes.begin(), window_sizes.end(), input_neuron_count_per_feature_map *= boost::lambda::_1);

		unsigned int current_weight_index = 0;
		for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_feature_map_count; ++output_feature_map_id)
		{
			unsigned int current_input_feature_map_count = data_custom[1][output_feature_map_id + 1] - data_custom[1][output_feature_map_id];
			if (current_input_feature_map_count > 0)
			{
				unsigned int input_neuron_count = input_neuron_count_per_feature_map * current_input_feature_map_count;
				float standard_deviation = 1.0F / sqrtf(static_cast<float>(input_neuron_count));
				float max_abs_value = 3.0F * standard_deviation;
				nnforge_normal_distribution<float> nd(0.0F, standard_deviation);

				for(unsigned int i = 0; i < input_neuron_count; ++i)
				{
					float val = nd(generator);
					while (fabs(val) > max_abs_value)
						val = nd(generator);

					data[0][current_weight_index] = val;
					++current_weight_index;
				}
			}
		}

		std::fill(data[1].begin(), data[1].end(), 0.0F);
	}

	void sparse_convolution_layer::randomize_custom_data(
		layer_data_custom& data_custom,
		random_generator& generator) const
	{
		std::vector<std::set<int> > out_fm_in_fm_present_list(output_feature_map_count, std::set<int>());

		std::vector<int> output_feature_map_id_list(feature_map_connection_count + output_feature_map_count);
		int i = 0;
		for(std::vector<int>::iterator it = output_feature_map_id_list.begin(); it != output_feature_map_id_list.end(); ++it)
		{
			*it = i;
			++i;
			if (i == output_feature_map_count)
				i = 0;
		}

		std::set<unsigned int> input_feature_map_id_set;
		unsigned int start_feature_map_index = 0;
		std::vector<int> v(input_feature_map_count);
		for(int i = 0; i < feature_map_connection_count; ++i)
		{
			if (input_feature_map_id_set.empty())
			{
				for(unsigned int input_feature_map_id = 0; input_feature_map_id < input_feature_map_count; ++input_feature_map_id)
					input_feature_map_id_set.insert(input_feature_map_id);
			}

			int current_feature_map_index = start_feature_map_index;
			for(; current_feature_map_index < output_feature_map_id_list.size(); ++current_feature_map_index)
			{
				if (output_feature_map_id_list[current_feature_map_index] == -1)
					continue;

				int output_feature_map_id = output_feature_map_id_list[current_feature_map_index];

				std::set<int>& in_fm_present_set = out_fm_in_fm_present_list[output_feature_map_id];

				std::vector<int>::iterator end_it = std::set_difference(
					input_feature_map_id_set.begin(),
					input_feature_map_id_set.end(),
					in_fm_present_set.begin(),
					in_fm_present_set.end(),
					v.begin());
				int count = end_it - v.begin();
				if (count == 0)
					continue;

				int input_feature_map_index = 0;
				if (count > 1)
				{
					nnforge_uniform_int_distribution<int> in_fm_dist(0U, count - 1);
					input_feature_map_index = in_fm_dist(generator);
				}
				int input_feature_map_id = v[input_feature_map_index];

				in_fm_present_set.insert(input_feature_map_id);
				input_feature_map_id_set.erase(input_feature_map_id);

				break;
			}
			if (current_feature_map_index == output_feature_map_id_list.size())
				throw neural_network_exception("Internal error when randomly initializing sparse connections");

			output_feature_map_id_list[current_feature_map_index] = -1;

			while (output_feature_map_id_list[start_feature_map_index] == -1)
			{
				++start_feature_map_index;
			}
		}

		int current_column_offset = 0;
		for(int output_feature_map_id = 0; output_feature_map_id < output_feature_map_count; ++output_feature_map_id)
		{
			data_custom[1][output_feature_map_id] = current_column_offset;
			const std::set<int>& input_feature_map_set = out_fm_in_fm_present_list[output_feature_map_id];
			std::copy(input_feature_map_set.begin(), input_feature_map_set.end(), data_custom[0].begin() + current_column_offset);

			current_column_offset += input_feature_map_set.size();
		}
		data_custom[1][output_feature_map_count] = current_column_offset;
	}

	float sparse_convolution_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count_per_feature_map();
		unsigned int per_item_flops = feature_map_connection_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);
		per_item_flops -= 1;

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float sparse_convolution_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count_per_feature_map();
		unsigned int per_item_flops = feature_map_connection_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float sparse_convolution_layer::get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific).get_neuron_count_per_feature_map();
		unsigned int per_item_flops = feature_map_connection_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	dropout_layer_config sparse_convolution_layer::get_dropout_layer_config(float dropout_rate) const
	{
		if ((dropout_rate < 0.0F) || (dropout_rate >= 1.0F))
			throw neural_network_exception((boost::format("Illegal dropout rate: %1%") % dropout_rate).str());

		dropout_layer_config res;
		res.weight_part_to_dropout_direct_multiplier_map.insert(std::make_pair<unsigned int, float>(0, 1.0F - dropout_rate));

		return res;
	}

	layer_data_configuration_list sparse_convolution_layer::get_layer_data_configuration_list() const
	{
		layer_data_configuration_list res;
		res.push_back(layer_data_configuration(1, feature_map_connection_count, window_sizes));
		res.push_back(layer_data_configuration(1, output_feature_map_count, std::vector<unsigned int>()));

		return res;
	}

	std::set<unsigned int> sparse_convolution_layer::get_weight_decay_part_id_set() const
	{
		std::set<unsigned int> res;
		res.insert(0);
		return res;
	}
}
