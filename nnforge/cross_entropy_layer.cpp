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

#include "cross_entropy_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	const std::string cross_entropy_layer::layer_type_name = "CrossEntropy";

	cross_entropy_layer::cross_entropy_layer(float scale)
		: scale(scale)
	{
	}

	std::string cross_entropy_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr cross_entropy_layer::clone() const
	{
		return layer::ptr(new cross_entropy_layer(*this));
	}

	layer_configuration cross_entropy_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if ((input_configuration_list[0].feature_map_count >= 0) && (input_configuration_list[1].feature_map_count >= 0) && (input_configuration_list[0].feature_map_count != input_configuration_list[1].feature_map_count))
			throw neural_network_exception((boost::format("Feature map counts in 2 input layers don't match: %1% and %2%") % input_configuration_list[0].feature_map_count % input_configuration_list[1].feature_map_count).str());

		return layer_configuration(1, input_configuration_list[0].dimension_count);
	}

	layer_configuration_specific cross_entropy_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count != input_configuration_specific_list[1].feature_map_count)
			throw neural_network_exception((boost::format("Feature map counts in 2 input layers don't match: %1% and %2%") % input_configuration_specific_list[0].feature_map_count % input_configuration_specific_list[1].feature_map_count).str());

		return layer_configuration_specific(1, input_configuration_specific_list[0].dimension_sizes);
	}

	bool cross_entropy_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}

	void cross_entropy_layer::write_proto(void * layer_proto) const
	{
		if (scale != 1.0F)
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::CrossEntropyParam * param = layer_proto_typed->mutable_cross_entropy_param();

			param->set_scale(scale);
		}
	}

	void cross_entropy_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_cross_entropy_param())
		{
			scale = 1.0F;
		}
		else
		{
			scale = layer_proto_typed->cross_entropy_param().scale();
		}
	}

	float cross_entropy_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = input_configuration_specific_list[0].feature_map_count * 8;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = input_configuration_specific_list[action.get_backprop_index()].get_neuron_count();
				unsigned int per_item_flops = input_configuration_specific_list[action.get_backprop_index()].feature_map_count * 6;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}
}
