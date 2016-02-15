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

#include "entry_convolution_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string entry_convolution_layer::layer_type_name = "EntryConvolution";

	entry_convolution_layer::entry_convolution_layer(unsigned int padding)
		: padding(padding)
	{
	}

	std::string entry_convolution_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr entry_convolution_layer::clone() const
	{
		return layer::ptr(new entry_convolution_layer(*this));
	}

	layer_configuration entry_convolution_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		layer_configuration res = input_configuration_list[0];
		if (res.feature_map_count >= 0)
			res.feature_map_count = res.feature_map_count * 2 - 1 + padding;
		return res;
	}

	layer_configuration_specific entry_convolution_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		layer_configuration_specific res = input_configuration_specific_list[0];
		res.feature_map_count = res.feature_map_count * 2 - 1 + padding;
		return res;
	}

	bool entry_convolution_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		layer_configuration_specific res = output_configuration_specific;
		res.feature_map_count = res.feature_map_count + 1 - padding;
		return true;
	}

	void entry_convolution_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::EntryConvolutionParam * param = layer_proto_typed->mutable_entry_convolution_param();

		if (padding > 0)
			param->set_padding(padding);
	}

	void entry_convolution_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_entry_convolution_param())
			throw neural_network_exception((boost::format("No entry_convolution_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::EntryConvolutionParam& param = layer_proto_typed->entry_convolution_param();

		padding = param.padding();
	}

	float entry_convolution_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				layer_configuration_specific output_config = get_output_layer_configuration_specific(input_configuration_specific_list);
				unsigned int neuron_count = output_config.get_neuron_count_per_feature_map();
				unsigned int per_item_flops = output_config.feature_map_count * output_config.feature_map_count;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				layer_configuration_specific output_config = get_output_layer_configuration_specific(input_configuration_specific_list);
				unsigned int neuron_count = output_config.get_neuron_count_per_feature_map();
				unsigned int per_item_flops = output_config.feature_map_count * output_config.feature_map_count;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}

	tiling_factor entry_convolution_layer::get_tiling_factor() const
	{
		return tiling_factor(2).get_inverse();
	}

	std::vector<std::string> entry_convolution_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		if (padding > 0)
			ss << "padding " << padding;

		res.push_back(ss.str());

		return res;
	}
}
