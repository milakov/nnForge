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

#include "cdf_max_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string cdf_max_layer::layer_type_name = "CDFMax";

	cdf_max_layer::cdf_max_layer(
		unsigned int entry_subsampling_size,
		bool is_min)
		: entry_subsampling_size(entry_subsampling_size)
		, is_min(is_min)
	{
		check();
	}

	void cdf_max_layer::check()
	{
		if (entry_subsampling_size < 2)
			throw neural_network_exception("entry subsampling size for cdf max layer may not be smaller than 2");
	}

	std::string cdf_max_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr cdf_max_layer::clone() const
	{
		return layer::ptr(new cdf_max_layer(*this));
	}

	void cdf_max_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::CDFMaxParam * param = layer_proto_typed->mutable_cdf_max_param();

		param->mutable_entry_param()->set_subsampling_size(entry_subsampling_size);

		if (is_min)
			param->set_function(nnforge::protobuf::CDFMaxParam_MaxFunction_MIN);
	}

	void cdf_max_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_cdf_max_param())
			throw neural_network_exception((boost::format("No cdf_max_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::CDFMaxParam& param = layer_proto_typed->cdf_max_param();

		if (!param.has_entry_param())
			throw neural_network_exception((boost::format("No entry_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		entry_subsampling_size = param.entry_param().subsampling_size();

		is_min = (param.function() == nnforge::protobuf::CDFMaxParam_MaxFunction_MIN);

		check();
	}

	float cdf_max_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = is_min ? entry_subsampling_size * 2 : entry_subsampling_size - 1;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = is_min ? entry_subsampling_size * 2 + 2 : entry_subsampling_size + 1;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}

	tiling_factor cdf_max_layer::get_tiling_factor() const
	{
		return tiling_factor(entry_subsampling_size).get_inverse();
	}

	std::vector<std::string> cdf_max_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		if (is_min)
			ss << "MIN, ";
		ss << "samples " << entry_subsampling_size;

		res.push_back(ss.str());

		return res;
	}
}
