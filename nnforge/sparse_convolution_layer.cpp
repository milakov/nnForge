/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "neural_network_exception.h"
#include "nn_types.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <set>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string sparse_convolution_layer::layer_type_name = "SparseConvolution";

	sparse_convolution_layer::sparse_convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count,
		unsigned int feature_map_connection_count,
		const std::vector<unsigned int>& left_zero_padding,
		const std::vector<unsigned int>& right_zero_padding)
		: window_sizes(window_sizes),
		input_feature_map_count(input_feature_map_count),
		output_feature_map_count(output_feature_map_count),
		feature_map_connection_sparsity_ratio(-1.0F),
		feature_map_connection_count(feature_map_connection_count),
		left_zero_padding(left_zero_padding),
		right_zero_padding(right_zero_padding)
	{
		if ((left_zero_padding.size() != 0) && (left_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for left zero padding") % left_zero_padding.size()).str());
		if ((right_zero_padding.size() != 0) && (right_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for right zero padding") % right_zero_padding.size()).str());

		if (left_zero_padding.empty())
			this->left_zero_padding.resize(window_sizes.size(), 0);
		else
			this->left_zero_padding = left_zero_padding;

		if (right_zero_padding.empty())
			this->right_zero_padding.resize(window_sizes.size(), 0);
		else
			this->right_zero_padding = right_zero_padding;

		check();
	}

	sparse_convolution_layer::sparse_convolution_layer(
		const std::vector<unsigned int>& window_sizes,
		unsigned int input_feature_map_count,
		unsigned int output_feature_map_count,
		float feature_map_connection_sparsity_ratio,
		const std::vector<unsigned int>& left_zero_padding,
		const std::vector<unsigned int>& right_zero_padding)
		: window_sizes(window_sizes),
		input_feature_map_count(input_feature_map_count),
		output_feature_map_count(output_feature_map_count),
		feature_map_connection_sparsity_ratio(feature_map_connection_sparsity_ratio),
		feature_map_connection_count(static_cast<unsigned int>(input_feature_map_count * output_feature_map_count * feature_map_connection_sparsity_ratio + 0.5F)),
		left_zero_padding(left_zero_padding),
		right_zero_padding(right_zero_padding)
	{
		if ((left_zero_padding.size() != 0) && (left_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for left zero padding") % left_zero_padding.size()).str());
		if ((right_zero_padding.size() != 0) && (right_zero_padding.size() != window_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for right zero padding") % right_zero_padding.size()).str());

		if (left_zero_padding.empty())
			this->left_zero_padding.resize(window_sizes.size(), 0);
		else
			this->left_zero_padding = left_zero_padding;

		if (right_zero_padding.empty())
			this->right_zero_padding.resize(window_sizes.size(), 0);
		else
			this->right_zero_padding = right_zero_padding;

		check();
	}

	void sparse_convolution_layer::check()
	{
		if (window_sizes.size() == 0)
			throw neural_network_exception("window sizes for sparse convolution layer may not be empty");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (window_sizes[i] == 0)
				throw neural_network_exception("window dimension for sparse convolution layer may not be zero");

		if (feature_map_connection_count < input_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be smaller than input_feature_map_count");
		if (feature_map_connection_count < output_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be smaller than output_feature_map_count");
		if (feature_map_connection_count > input_feature_map_count * output_feature_map_count)
			throw neural_network_exception("feature_map_connection_count may not be larger than in dense case");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (left_zero_padding[i] >= window_sizes[i])
				throw neural_network_exception((boost::format("left zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % left_zero_padding[i] % i % window_sizes[i]).str());

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (right_zero_padding[i] >= window_sizes[i])
				throw neural_network_exception((boost::format("right zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % right_zero_padding[i] % i % window_sizes[i]).str());
	}

	std::string sparse_convolution_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr sparse_convolution_layer::clone() const
	{
		return layer::ptr(new sparse_convolution_layer(*this));
	}

	layer_configuration sparse_convolution_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if ((input_configuration_list[0].feature_map_count >= 0) && (input_configuration_list[0].feature_map_count != static_cast<int>(input_feature_map_count)))
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration_list[0].feature_map_count).str());

		if ((input_configuration_list[0].dimension_count >= 0) && (input_configuration_list[0].dimension_count != static_cast<int>(window_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration_list[0].dimension_count).str());

		return layer_configuration(output_feature_map_count, static_cast<int>(window_sizes.size()));
	}

	layer_configuration_specific sparse_convolution_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count != input_feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration_specific_list[0].feature_map_count).str());

		if (input_configuration_specific_list[0].get_dimension_count() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res(output_feature_map_count);

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
		{
			unsigned int total_input_dimension_size = input_configuration_specific_list[0].dimension_sizes[i] + left_zero_padding[i] + right_zero_padding[i];
			if (total_input_dimension_size < window_sizes[i])
				throw neural_network_exception((boost::format("Too small total dimension size (with padding) %1% of dimension (%2%) is smaller than layer window size (%3%)") % total_input_dimension_size % i % window_sizes[i]).str());

			res.dimension_sizes.push_back(total_input_dimension_size + 1 - window_sizes[i]);
		}

		return res;
	}

	bool sparse_convolution_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.feature_map_count != output_feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and output configuration (%2%) don't match") % output_feature_map_count % output_configuration_specific.feature_map_count).str());

		if (output_configuration_specific.get_dimension_count() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % output_configuration_specific.get_dimension_count()).str());

		input_configuration_specific = layer_configuration_specific(output_feature_map_count);

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
			input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] + window_sizes[i] - 1 - left_zero_padding[i] - right_zero_padding[i]);

		return true;
	}

	std::vector<std::pair<unsigned int, unsigned int> > sparse_convolution_layer::get_input_rectangle_borders(
		const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		unsigned int input_layer_id) const
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

	void sparse_convolution_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::SparseConvolutionalParam * param = layer_proto_typed->mutable_sparse_convolution_param();

		param->set_output_feature_map_count(output_feature_map_count);
		param->set_input_feature_map_count(input_feature_map_count);

		if (feature_map_connection_sparsity_ratio >= 0.0F)
			param->set_feature_map_connection_sparsity_ratio(feature_map_connection_sparsity_ratio);
		else
			param->set_feature_map_connection_count(feature_map_connection_count);

		for(int i = 0; i < window_sizes.size(); ++i)
		{
			protobuf::SparseConvolutionalParam_SparseConvolutionalDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_kernel_size(window_sizes[i]);
			if (left_zero_padding[i] > 0)
				dim_param->set_left_padding(left_zero_padding[i]);
			if (right_zero_padding[i] > 0)
				dim_param->set_right_padding(right_zero_padding[i]);
		}
	}

	void sparse_convolution_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_sparse_convolution_param())
			throw neural_network_exception((boost::format("No sparse_convolution_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		input_feature_map_count = layer_proto_typed->sparse_convolution_param().input_feature_map_count();
		output_feature_map_count = layer_proto_typed->sparse_convolution_param().output_feature_map_count();

		window_sizes.resize(layer_proto_typed->sparse_convolution_param().dimension_param_size());
		left_zero_padding.resize(layer_proto_typed->sparse_convolution_param().dimension_param_size());
		right_zero_padding.resize(layer_proto_typed->sparse_convolution_param().dimension_param_size());

		for(int i = 0; i < layer_proto_typed->sparse_convolution_param().dimension_param_size(); ++i)
		{
			window_sizes[i] = layer_proto_typed->sparse_convolution_param().dimension_param(i).kernel_size();
			left_zero_padding[i] = layer_proto_typed->sparse_convolution_param().dimension_param(i).left_padding();
			right_zero_padding[i] = layer_proto_typed->sparse_convolution_param().dimension_param(i).right_padding();
		}

		if (layer_proto_typed->sparse_convolution_param().has_feature_map_connection_count())
		{
			feature_map_connection_sparsity_ratio = -1.0F;
			feature_map_connection_count = layer_proto_typed->sparse_convolution_param().feature_map_connection_count();
		}
		else if (layer_proto_typed->sparse_convolution_param().has_feature_map_connection_sparsity_ratio())
		{
			feature_map_connection_sparsity_ratio = layer_proto_typed->sparse_convolution_param().feature_map_connection_sparsity_ratio();
			feature_map_connection_count = static_cast<unsigned int>(input_feature_map_count * output_feature_map_count * feature_map_connection_sparsity_ratio + 0.5F);
		}
		else
			throw neural_network_exception((boost::format("No sparsity pattern defined in sparse_convolution_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		check();
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
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
		randomize_custom_data(*data_custom, generator);

		randomize_weights(*data, *data_custom, generator);
	}

	void sparse_convolution_layer::randomize_weights(
		layer_data& data,
		const layer_data_custom& data_custom,
		random_generator& generator) const
	{
		unsigned int weight_count = 1;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);

		unsigned int current_weight_index = 0;
		for(unsigned int output_feature_map_id = 0; output_feature_map_id < output_feature_map_count; ++output_feature_map_id)
		{
			unsigned int current_input_feature_map_count = data_custom[1][output_feature_map_id + 1] - data_custom[1][output_feature_map_id];
			if (current_input_feature_map_count > 0)
			{
				float current_average_feature_map_count = sqrtf(static_cast<float>(current_input_feature_map_count) * static_cast<float>(output_feature_map_count));
				float standard_deviation = sqrtf(1.0F / (current_average_feature_map_count * static_cast<float>(weight_count)));
				float max_abs_value = 100.0F * standard_deviation;
				nnforge_normal_distribution<float> nd(0.0F, standard_deviation);

				unsigned int currrent_input_neuron_count = weight_count * current_input_feature_map_count;
				for(unsigned int i = 0; i < currrent_input_neuron_count; ++i)
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
		for(int i = 0; i < static_cast<int>(feature_map_connection_count); ++i)
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
				int count = static_cast<int>(end_it - v.begin());
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
		for(int output_feature_map_id = 0; output_feature_map_id < static_cast<int>(output_feature_map_count); ++output_feature_map_id)
		{
			data_custom[1][output_feature_map_id] = current_column_offset;
			const std::set<int>& input_feature_map_set = out_fm_in_fm_present_list[output_feature_map_id];
			std::copy(input_feature_map_set.begin(), input_feature_map_set.end(), data_custom[0].begin() + current_column_offset);

			current_column_offset += static_cast<int>(input_feature_map_set.size());
		}
		data_custom[1][output_feature_map_count] = current_column_offset;
	}

	float sparse_convolution_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
		case layer_action::backward_data:
		case layer_action::backward_weights:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count_per_feature_map();
				unsigned int per_item_flops = feature_map_connection_count * 2;
				std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);
				per_item_flops -= 1;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
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

	std::vector<std::string> sparse_convolution_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		if (window_sizes.empty())
		{
			ss << "fc";
		}
		else
		{
			for(int i = 0; i < window_sizes.size(); ++i)
			{
				if (i != 0)
					ss << "x";
				ss << window_sizes[i];
			}
		}
		ss << ", fm " << input_feature_map_count << "x" << output_feature_map_count;

		bool empty_padding = true;
		for(int i = 0; i < left_zero_padding.size(); ++i)
		{
			if ((left_zero_padding[i] != 0) || (right_zero_padding[i] != 0))
			{
				empty_padding = false;
				break;
			}
		}
		if (!empty_padding)
		{
			ss << ", pad ";
			for(int i = 0; i < left_zero_padding.size(); ++i)
			{
				if (i != 0)
					ss << "x";
				if (left_zero_padding[i] == right_zero_padding[i])
					ss << left_zero_padding[i];
				else
					ss << left_zero_padding[i] << "_" << right_zero_padding[i];
			}
		}

		res.push_back(ss.str());
		ss.clear();

		if (feature_map_connection_sparsity_ratio >= 0.0F)
			ss << (boost::format("fm connection ratio %|1$.5f|") % feature_map_connection_sparsity_ratio).str();
		else
			ss << "fm connections " << feature_map_connection_count;

		res.push_back(ss.str());

		return res;
	}
}
