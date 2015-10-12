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
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <numeric>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <opencv2/core/core.hpp>

namespace nnforge
{
	const std::string convolution_layer::layer_type_name = "Convolution";

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

	void convolution_layer::check()
	{
		if (window_sizes.size() == 0)
			throw neural_network_exception("window sizes for convolution layer may not be empty");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (window_sizes[i] == 0)
				throw neural_network_exception("window dimension for convolution layer may not be zero");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (left_zero_padding[i] >= window_sizes[i])
				throw neural_network_exception((boost::format("left zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % left_zero_padding[i] % i % window_sizes[i]).str());

		for(unsigned int i = 0; i < window_sizes.size(); i++)
			if (right_zero_padding[i] >= window_sizes[i])
				throw neural_network_exception((boost::format("right zero padding %1% of dimension (%2%) is greater or equal than layer window size (%3%)") % right_zero_padding[i] % i % window_sizes[i]).str());
	}

	std::string convolution_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr convolution_layer::clone() const
	{
		return layer::ptr(new convolution_layer(*this));
	}

	layer_configuration convolution_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if ((input_configuration_list[0].feature_map_count >= 0) && (input_configuration_list[0].feature_map_count != static_cast<int>(input_feature_map_count)))
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % input_feature_map_count % input_configuration_list[0].feature_map_count).str());

		if ((input_configuration_list[0].dimension_count >= 0) && (input_configuration_list[0].dimension_count != static_cast<int>(window_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration_list[0].dimension_count).str());

		return layer_configuration(output_feature_map_count, static_cast<int>(window_sizes.size()));
	}

	layer_configuration_specific convolution_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
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

	layer_configuration_specific convolution_layer::get_input_layer_configuration_specific(
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.feature_map_count != output_feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and output configuration (%2%) don't match") % output_feature_map_count % output_configuration_specific.feature_map_count).str());

		if (output_configuration_specific.get_dimension_count() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % window_sizes.size() % output_configuration_specific.get_dimension_count()).str());

		layer_configuration_specific res(input_feature_map_count);

		for(unsigned int i = 0; i < window_sizes.size(); ++i)
			res.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] + window_sizes[i] - 1 - left_zero_padding[i] - right_zero_padding[i]);

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > convolution_layer::get_input_rectangle_borders(
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

	void convolution_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::ConvolutionalParam * param = layer_proto_typed->mutable_convolution_param();

		param->set_output_feature_map_count(output_feature_map_count);
		param->set_input_feature_map_count(input_feature_map_count);

		for(int i = 0; i < window_sizes.size(); ++i)
		{
			protobuf::ConvolutionalParam_ConvolutionalDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_kernel_size(window_sizes[i]);
			if (left_zero_padding[i] > 0)
				dim_param->set_left_padding(left_zero_padding[i]);
			if (right_zero_padding[i] > 0)
				dim_param->set_right_padding(right_zero_padding[i]);
		}
	}

	void convolution_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_convolution_param())
			throw neural_network_exception((boost::format("No convolution_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		input_feature_map_count = layer_proto_typed->convolution_param().input_feature_map_count();
		output_feature_map_count = layer_proto_typed->convolution_param().output_feature_map_count();

		window_sizes.resize(layer_proto_typed->convolution_param().dimension_param_size());
		left_zero_padding.resize(layer_proto_typed->convolution_param().dimension_param_size());
		right_zero_padding.resize(layer_proto_typed->convolution_param().dimension_param_size());

		for(int i = 0; i < layer_proto_typed->convolution_param().dimension_param_size(); ++i)
		{
			window_sizes[i] = layer_proto_typed->convolution_param().dimension_param(i).kernel_size();
			left_zero_padding[i] = layer_proto_typed->convolution_param().dimension_param(i).left_padding();
			right_zero_padding[i] = layer_proto_typed->convolution_param().dimension_param(i).right_padding();
		}

		check();
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
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
		unsigned int weight_count = 1;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);

		float average_feature_map_count = sqrtf(static_cast<float>(input_feature_map_count) * static_cast<float>(output_feature_map_count));

		float standard_deviation = sqrtf(1.0F / (average_feature_map_count * static_cast<float>(weight_count)));
		float max_abs_value = 100.0F * standard_deviation;

		nnforge_normal_distribution<float> nd(0.0F, standard_deviation);

		for(unsigned int i = 0; i < (*data)[0].size(); ++i)
		{
			float val = nd(generator);
			while (fabs(val) > max_abs_value)
				val = nd(generator);

			(*data)[0][i] = val;
		}

		std::fill((*data)[1].begin(), (*data)[1].end(), 0.0F);
	}

	void convolution_layer::randomize_orthogonal_data(
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
		unsigned int weight_count = 1;
		std::for_each(window_sizes.begin(), window_sizes.end(), weight_count *= boost::lambda::_1);
		unsigned int weight_col_count = weight_count * input_feature_map_count;
		unsigned int weight_row_count = output_feature_map_count;

		nnforge_normal_distribution<float> nd(0.0F, 1.0F);
		for(unsigned int i = 0; i < (*data)[0].size(); ++i)
		{
			float val = nd(generator);
			(*data)[0][i] = val;
		}
		cv::Mat1f weights(weight_row_count, weight_col_count, &((*data)[0][0]));
		cv::Mat1f w;
		cv::Mat1f u;
		cv::Mat1f vt;
		cv::SVD::compute(weights, w, u, vt, cv::SVD::MODIFY_A);

		cv::Mat1f orth;
		if ((u.rows == weights.rows) && (u.cols == weights.cols))
			orth = u;
		else if ((vt.rows == weights.rows) && (vt.cols == weights.cols))
			orth = vt;
		else
			throw neural_network_exception("Internal error when doing SVD");

		std::copy(orth.begin(), orth.end(), weights.begin());

		std::fill((*data)[1].begin(), (*data)[1].end(), 0.0F);
	}

	float convolution_layer::get_forward_flops(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);
		per_item_flops -= 1;

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_backward_flops(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		unsigned int input_layer_id) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
		unsigned int per_item_flops = input_feature_map_count * 2;
		std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops *= boost::lambda::_1);

		return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
	}

	float convolution_layer::get_weights_update_flops(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
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
