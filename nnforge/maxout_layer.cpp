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

#include "maxout_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string maxout_layer::layer_type_name = "Maxout";

	maxout_layer::maxout_layer(unsigned int feature_map_subsampling_size)
		: feature_map_subsampling_size(feature_map_subsampling_size)
	{
		check();
	}

	void maxout_layer::check()
	{
		if (feature_map_subsampling_size < 2)
			throw neural_network_exception("Feature map subsampling size should be >= 2 for maxout layer");
	}

	std::string maxout_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr maxout_layer::clone() const
	{
		return layer::ptr(new maxout_layer(*this));
	}

	layer_configuration_specific maxout_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if ((input_configuration_specific_list[0].feature_map_count % feature_map_subsampling_size) != 0)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not evenly divisible by feature map subsampling count (%2%)") % input_configuration_specific_list[0].feature_map_count % feature_map_subsampling_size).str());

		return layer_configuration_specific(input_configuration_specific_list[0].feature_map_count / feature_map_subsampling_size, input_configuration_specific_list[0].dimension_sizes);
	}

	bool maxout_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		input_configuration_specific = layer_configuration_specific(output_configuration_specific.feature_map_count * feature_map_subsampling_size, output_configuration_specific.dimension_sizes);

		return true;
	}

	void maxout_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::MaxoutParam * param = layer_proto_typed->mutable_maxout_param();

		param->set_feature_map_subsampling_size(feature_map_subsampling_size);
	}

	void maxout_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_maxout_param())
			throw neural_network_exception((boost::format("No maxout_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		feature_map_subsampling_size = layer_proto_typed->maxout_param().feature_map_subsampling_size();

		check();
	}

	float maxout_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = feature_map_subsampling_size - 1;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			return 0.0F;
		default:
			return 0.0F;
		}
	}

	std::vector<std::string> maxout_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		ss << "fm subsampling " << feature_map_subsampling_size;

		res.push_back(ss.str());

		return res;
	}
}
