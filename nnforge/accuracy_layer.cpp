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

#include "accuracy_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string accuracy_layer::layer_type_name = "Accuracy";

	accuracy_layer::accuracy_layer(unsigned int top_n)
		: top_n(top_n)
	{
	}

	std::string accuracy_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr accuracy_layer::clone() const
	{
		return layer::ptr(new accuracy_layer(*this));
	}

	layer_configuration_specific accuracy_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count != input_configuration_specific_list[1].feature_map_count)
			throw neural_network_exception((boost::format("Feature map counts in 2 input layers for accuracy_layer don't match: %1% and %2%") % input_configuration_specific_list[0].feature_map_count % input_configuration_specific_list[1].feature_map_count).str());

		if (input_configuration_specific_list[0].get_neuron_count_per_feature_map() != input_configuration_specific_list[1].get_neuron_count_per_feature_map())
			throw neural_network_exception("Neuron count per feature maps mismatch in 2 input layers for accuracy_layer");

		if (input_configuration_specific_list.size() > 2)
		{
			if (input_configuration_specific_list[2].feature_map_count != 1)
				throw neural_network_exception((boost::format("Feature map count for accuracy_layer scaling should be equal to 1, while it is %1%") % input_configuration_specific_list[2].feature_map_count).str());

			if (input_configuration_specific_list[2].get_neuron_count_per_feature_map() != input_configuration_specific_list[0].get_neuron_count_per_feature_map())
				throw neural_network_exception((boost::format("Neuron count per feature map accuracy_layer for scaling equals %1%, expected %2%") % input_configuration_specific_list[2].get_neuron_count_per_feature_map() % input_configuration_specific_list[0].get_neuron_count_per_feature_map()).str());
		}

		return layer_configuration_specific(top_n + 1, input_configuration_specific_list[0].dimension_sizes);
	}

	bool accuracy_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}

	void accuracy_layer::write_proto(void * layer_proto) const
	{
		if (top_n != 1)
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::AccuracyParam * param = layer_proto_typed->mutable_accuracy_param();

			param->set_top_n(top_n);
		}
	}

	void accuracy_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_accuracy_param())
		{
			top_n = 1;
		}
		else
		{
			top_n = layer_proto_typed->accuracy_param().top_n();
		}
	}

	float accuracy_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = input_configuration_specific_list[0].feature_map_count * 2;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			throw neural_network_exception("get_backward_flops is not implemented for accuracy_layer");
		default:
			return 0.0F;
		}
	}

	std::string accuracy_layer::get_string_for_average_data(
		const layer_configuration_specific& config,
		const std::vector<double>& data) const
	{
		std::stringstream s;
		s << instance_name << " acc/err = ";
		std::vector<double>::const_iterator it = data.begin();
		float scale = 1.0F / static_cast<float>(data.back());
		for(unsigned int top_i_index = 0; top_i_index < top_n; ++it, ++top_i_index)
		{
			if (it != data.begin())
				s << ", ";
			float acc_val = static_cast<float>(*it) * scale * 100.0F;
			s << (boost::format("Top-%1% %|2$.2f|%%/%|3$.2f|%%") % (top_i_index + 1) % acc_val % (100.0F - acc_val)).str();
		}
		return s.str();
	}

	std::vector<std::string> accuracy_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		ss << "top " << top_n;

		res.push_back(ss.str());

		return res;
	}
}
