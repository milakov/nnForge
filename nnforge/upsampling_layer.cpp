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

#include "upsampling_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string upsampling_layer::layer_type_name = "Upsampling";

	upsampling_layer::upsampling_layer(
		const std::vector<unsigned int>& upsampling_sizes,
		unsigned int feature_map_upsampling_size,
		unsigned int entry_upsampling_size)
		: upsampling_sizes(upsampling_sizes)
		, feature_map_upsampling_size(feature_map_upsampling_size)
		, entry_upsampling_size(entry_upsampling_size)
	{
		check();
	}

	void upsampling_layer::check()
	{
		for(unsigned int i = 0; i < upsampling_sizes.size(); i++)
		{
			if (upsampling_sizes[i] == 0)
				throw neural_network_exception("window dimension for upsampling layer may not be zero");
		}

		if (feature_map_upsampling_size == 0)
			throw neural_network_exception("feature map upsampling size for upsampling layer may not be zero");

		if (entry_upsampling_size == 0)
			throw neural_network_exception("feature map upsampling size for upsampling layer may not be zero");
	}

	std::string upsampling_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr upsampling_layer::clone() const
	{
		return layer::ptr(new upsampling_layer(*this));
	}

	layer_configuration upsampling_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if ((input_configuration_list[0].dimension_count >= 0) && (input_configuration_list[0].dimension_count != static_cast<int>(upsampling_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % upsampling_sizes.size() % input_configuration_list[0].dimension_count).str());

		int new_feature_map_count = input_configuration_list[0].feature_map_count;
		if (new_feature_map_count >= 0)
			new_feature_map_count *= feature_map_upsampling_size;

		return layer_configuration(new_feature_map_count, static_cast<int>(upsampling_sizes.size()));
	}

	layer_configuration_specific upsampling_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_dimension_count() != upsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % upsampling_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific_list[0].feature_map_count * feature_map_upsampling_size);

		for(unsigned int i = 0; i < upsampling_sizes.size(); ++i)
			res.dimension_sizes.push_back(input_configuration_specific_list[0].dimension_sizes[i] * upsampling_sizes[i]);

		return res;
	}

	bool upsampling_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.get_dimension_count() != upsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % upsampling_sizes.size() % output_configuration_specific.get_dimension_count()).str());

		if (output_configuration_specific.feature_map_count % feature_map_upsampling_size != 0)
			throw neural_network_exception((boost::format("Feature map count in output config (%1%) is not evenly divisible by feature map upsampling size (%2%)") % output_configuration_specific.feature_map_count % feature_map_upsampling_size).str());

		input_configuration_specific = layer_configuration_specific(output_configuration_specific.feature_map_count / feature_map_upsampling_size);

		for(unsigned int i = 0; i < upsampling_sizes.size(); ++i)
		{
			if (output_configuration_specific.feature_map_count % feature_map_upsampling_size != 0)
				throw neural_network_exception((boost::format("Dimension size in output config (%1%) is not evenly divisible by upsampling size (%2%)") % output_configuration_specific.dimension_sizes[i] % upsampling_sizes[i]).str());

			input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] / upsampling_sizes[i]);
		}

		return true;
	}

	void upsampling_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<nnforge::protobuf::Layer *>(layer_proto);
		nnforge::protobuf::UpsamplingParam * param = layer_proto_typed->mutable_upsampling_param();
		for(int i = 0; i < upsampling_sizes.size(); ++i)
		{
			nnforge::protobuf::UpsamplingParam_UpsamplingDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_upsampling_size(upsampling_sizes[i]);
		}

		if (feature_map_upsampling_size != 1)
			param->mutable_feature_map_param()->set_upsampling_size(feature_map_upsampling_size);

		if (entry_upsampling_size != 1)
			param->mutable_entry_param()->set_upsampling_size(entry_upsampling_size);
	}

	void upsampling_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const nnforge::protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_upsampling_param())
			throw neural_network_exception((boost::format("No upsampling_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::UpsamplingParam& param = layer_proto_typed->upsampling_param();

		upsampling_sizes.resize(param.dimension_param_size());
		for(int i = 0; i < param.dimension_param_size(); ++i)
			upsampling_sizes[i] = param.dimension_param(i).upsampling_size();

		feature_map_upsampling_size = param.has_feature_map_param() ? param.feature_map_param().upsampling_size() : 1;

		entry_upsampling_size = param.has_entry_param() ? param.entry_param().upsampling_size() : 1;

		check();
	}

	float upsampling_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return 0.0F;
		case layer_action::backward_data:
			{
				unsigned int neuron_count = input_configuration_specific_list[0].get_neuron_count();
				unsigned int per_item_flops = feature_map_upsampling_size;
				std::for_each(upsampling_sizes.begin(), upsampling_sizes.end(), per_item_flops *= boost::lambda::_1);
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}

	tiling_factor upsampling_layer::get_tiling_factor() const
	{
		return tiling_factor(entry_upsampling_size);
	}

	std::vector<std::string> upsampling_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		for(int i = 0; i < upsampling_sizes.size(); ++i)
		{
			if (i != 0)
				ss << "x";
			ss << upsampling_sizes[i];
		}

		if (feature_map_upsampling_size != 1)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "fm " << feature_map_upsampling_size;
		}

		if (entry_upsampling_size != 1)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "samples " << entry_upsampling_size;
		}

		res.push_back(ss.str());

		return res;
	}
}
