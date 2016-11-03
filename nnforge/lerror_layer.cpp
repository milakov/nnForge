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

#include "lerror_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string lerror_layer::layer_type_name = "LError";

	lerror_layer::lerror_layer(
		float n,
		float scale)
		: n(n)
		, scale(scale)
	{
	}

	std::string lerror_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr lerror_layer::clone() const
	{
		return layer::ptr(new lerror_layer(*this));
	}

	layer_configuration_specific lerror_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count != input_configuration_specific_list[1].feature_map_count)
			throw neural_network_exception((boost::format("Feature map counts in 2 input layers for mse_layer don't match: %1% and %2%") % input_configuration_specific_list[0].feature_map_count % input_configuration_specific_list[1].feature_map_count).str());

		if (input_configuration_specific_list[0].get_neuron_count_per_feature_map() != input_configuration_specific_list[1].get_neuron_count_per_feature_map())
			throw neural_network_exception("Neuron count per feature maps mismatch in 2 input layers for mse_layer");

		if (input_configuration_specific_list.size() > 2)
		{
			if (input_configuration_specific_list[2].feature_map_count != 1)
				throw neural_network_exception((boost::format("Feature map count for mse_layer scaling should be equal to 1, while it is %1%") % input_configuration_specific_list[2].feature_map_count).str());

			if (input_configuration_specific_list[2].get_neuron_count_per_feature_map() != input_configuration_specific_list[0].get_neuron_count_per_feature_map())
				throw neural_network_exception((boost::format("Neuron count per feature map mse_layer for scaling equals %1%, expected %2%") % input_configuration_specific_list[2].get_neuron_count_per_feature_map() % input_configuration_specific_list[0].get_neuron_count_per_feature_map()).str());
		}

		return layer_configuration_specific(1, input_configuration_specific_list[0].dimension_sizes);
	}

	bool lerror_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}

	void lerror_layer::write_proto(void * layer_proto) const
	{
		if ((n != 2.0F) && (scale != 1.0F))
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::LErrorParam * param = layer_proto_typed->mutable_lerror_param();

			if (n != 2.0F)
				param->set_n(n);
			if (scale != 1.0F)
				param->set_scale(scale);
		}
	}

	void lerror_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_lerror_param())
		{
			n = 2.0F;
			scale = 1.0F;
		}
		else
		{
			n = layer_proto_typed->lerror_param().n();
			scale = layer_proto_typed->lerror_param().scale();
		}
	}

	float lerror_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = input_configuration_specific_list[0].feature_map_count * 3;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = input_configuration_specific_list[action.get_backprop_index()].get_neuron_count();
				unsigned int per_item_flops = input_configuration_specific_list[action.get_backprop_index()].feature_map_count * 2;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}

	std::vector<std::string> lerror_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		ss << "L" << n;
		if (scale != 1.0F)
			ss << ", scale " << scale;

		res.push_back(ss.str());

		return res;
	}
}
