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

#include "average_subsampling_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string average_subsampling_layer::layer_type_name = "AverageSubsampling";

	average_subsampling_layer::average_subsampling_layer(
		const std::vector<average_subsampling_factor>& subsampling_sizes,
		average_subsampling_factor feature_map_subsampling_size,
		unsigned int entry_subsampling_size,
		float alpha)
		: subsampling_sizes(subsampling_sizes)
		, feature_map_subsampling_size(feature_map_subsampling_size)
		, entry_subsampling_size(entry_subsampling_size)
		, alpha(alpha)
	{
		check();
	}

	void average_subsampling_layer::check()
	{
		for(unsigned int i = 0; i < subsampling_sizes.size(); i++)
		{
			if (subsampling_sizes[i].get_factor() == 0)
				throw neural_network_exception("window dimension for average subsampling layer may not be zero");
		}

		if (feature_map_subsampling_size.get_factor() == 0)
			throw neural_network_exception("feature map subsampling size for average subsampling layer may not be zero");

		if (entry_subsampling_size == 0)
			throw neural_network_exception("feature map subsampling size for average subsampling layer may not be zero");
	}

	std::string average_subsampling_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr average_subsampling_layer::clone() const
	{
		return layer::ptr(new average_subsampling_layer(*this));
	}

	layer_configuration_specific average_subsampling_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res;
		if (feature_map_subsampling_size.is_relative())
			res.feature_map_count = input_configuration_specific_list[0].feature_map_count / feature_map_subsampling_size.get_factor();
		else
		{
			res.feature_map_count = feature_map_subsampling_size.get_factor();
			get_fm_subsampling_size(input_configuration_specific_list[0].feature_map_count, res.feature_map_count);
		}

		for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
		{
			if (subsampling_sizes[i].is_relative())
			{
				if (input_configuration_specific_list[0].dimension_sizes[i] < subsampling_sizes[i].get_factor())
					throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than subsampling size (%3%)") % input_configuration_specific_list[0].dimension_sizes[i] % i % subsampling_sizes[i].get_factor()).str());
				res.dimension_sizes.push_back(input_configuration_specific_list[0].dimension_sizes[i] / subsampling_sizes[i].get_factor());
			}
			else
			{
				if (input_configuration_specific_list[0].dimension_sizes[i] < subsampling_sizes[i].get_factor())
					throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than subsampled size (%3%)") % input_configuration_specific_list[0].dimension_sizes[i] % i % subsampling_sizes[i].get_factor()).str());
				unsigned int output_size = subsampling_sizes[i].get_factor();
				get_subsampling_size(i, input_configuration_specific_list[0].dimension_sizes[i], output_size);
				res.dimension_sizes.push_back(output_size);
			}
		}

		return res;
	}

	unsigned int average_subsampling_layer::get_subsampling_size(
		unsigned int dimension_id,
		unsigned int input,
		unsigned int output) const
	{
		if (subsampling_sizes[dimension_id].is_relative())
			return subsampling_sizes[dimension_id].get_factor();

		unsigned int factor = input / output;
		unsigned int new_output = input / factor;
		if (new_output != output)
			throw neural_network_exception((boost::format("Cannot calculate subsampling factor for input %1% and output %2%") % input % output).str());

		return factor;
	}

	unsigned int average_subsampling_layer::get_fm_subsampling_size(
		unsigned int input,
		unsigned int output) const
	{
		if (feature_map_subsampling_size.is_relative())
			return feature_map_subsampling_size.get_factor();

		unsigned int factor = input / output;
		unsigned int new_output = input / factor;
		if (new_output != output)
			throw neural_network_exception((boost::format("Cannot calculate fm subsampling factor for input %1% and output %2%") % input % output).str());

		return factor;
	}

	bool average_subsampling_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % subsampling_sizes.size() % output_configuration_specific.get_dimension_count()).str());

		input_configuration_specific = layer_configuration_specific(output_configuration_specific.feature_map_count * (feature_map_subsampling_size.is_relative() ? feature_map_subsampling_size.get_factor() : 1U));

		for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
			input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] * (subsampling_sizes[i].is_relative() ? subsampling_sizes[i].get_factor() : 1U));

		return true;
	}

	void average_subsampling_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<nnforge::protobuf::Layer *>(layer_proto);
		nnforge::protobuf::AverageSubsamplingParam * param = layer_proto_typed->mutable_average_subsampling_param();
		for(int i = 0; i < subsampling_sizes.size(); ++i)
		{
			nnforge::protobuf::AverageSubsamplingParam_AverageSubsamplingDimensionParam * dim_param = param->add_dimension_param();
			if (subsampling_sizes[i].is_relative())
				dim_param->set_subsampling_size(subsampling_sizes[i].get_factor());
			else
				dim_param->set_subsampled_size(subsampling_sizes[i].get_factor());
		}

		if (!feature_map_subsampling_size.is_relative() || (feature_map_subsampling_size.get_factor() != 1))
		{
			if (feature_map_subsampling_size.is_relative())
				param->mutable_feature_map_param()->set_subsampling_size(feature_map_subsampling_size.get_factor());
			else
				param->mutable_feature_map_param()->set_subsampled_size(feature_map_subsampling_size.get_factor());
		}

		if (entry_subsampling_size != 1)
			param->mutable_entry_param()->set_subsampling_size(entry_subsampling_size);

		if (alpha != -std::numeric_limits<float>::max())
			param->set_alpha(alpha);
	}

	void average_subsampling_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const nnforge::protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_average_subsampling_param())
			throw neural_network_exception((boost::format("No average_subsampling_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::AverageSubsamplingParam& param = layer_proto_typed->average_subsampling_param();

		subsampling_sizes.resize(param.dimension_param_size());
		for(int i = 0; i < param.dimension_param_size(); ++i)
		{
			if (param.dimension_param(i).has_subsampling_size())
				subsampling_sizes[i] = average_subsampling_factor(param.dimension_param(i).subsampling_size(), true);
			else if (param.dimension_param(i).has_subsampled_size())
				subsampling_sizes[i] = average_subsampling_factor(param.dimension_param(i).subsampled_size(), false);
			else
				throw neural_network_exception("Neither subsampling_size nor subsampled_size specified for dimension_param of average_subsampling_layer");
		}

		if (param.has_feature_map_param())
		{
			if (param.feature_map_param().has_subsampling_size())
				feature_map_subsampling_size = average_subsampling_factor(param.feature_map_param().subsampling_size(), true);
			else if (param.feature_map_param().has_subsampled_size())
				feature_map_subsampling_size = average_subsampling_factor(param.feature_map_param().subsampled_size(), false);
			else
				throw neural_network_exception("Neither subsampling_size nor subsampled_size specified for feature_map_param of average_subsampling_layer");
		}
		else
			feature_map_subsampling_size = average_subsampling_factor(1, true);

		entry_subsampling_size = param.has_entry_param() ? param.entry_param().subsampling_size() : 1;

		alpha = param.has_alpha() ? param.alpha() : -std::numeric_limits<float>::max();

		check();
	}

	float average_subsampling_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				layer_configuration_specific output_config = get_output_layer_configuration_specific(input_configuration_specific_list);
				unsigned int neuron_count = output_config.get_neuron_count();
				unsigned int per_item_flops = get_fm_subsampling_size(input_configuration_specific_list[0].feature_map_count, output_config.feature_map_count) * entry_subsampling_size;
				for(unsigned int i = 0; i < static_cast<unsigned int>(output_config.dimension_sizes.size()); ++i)
					per_item_flops *= get_subsampling_size(i, input_configuration_specific_list[0].dimension_sizes[i], output_config.dimension_sizes[i]);
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				return static_cast<float>(neuron_count);
			}
		default:
			return 0.0F;
		}
	}

	tiling_factor average_subsampling_layer::get_tiling_factor() const
	{
		return tiling_factor(entry_subsampling_size).get_inverse();
	}

	std::vector<std::string> average_subsampling_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		for(int i = 0; i < subsampling_sizes.size(); ++i)
		{
			if (i != 0)
				ss << "x";
			ss << subsampling_sizes[i].get_factor();
			if (!subsampling_sizes[i].is_relative())
				ss << "_abs_";
		}

		if ((!feature_map_subsampling_size.is_relative()) || (feature_map_subsampling_size.get_factor() != 1))
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "fm " << feature_map_subsampling_size.get_factor();
			if (!feature_map_subsampling_size.is_relative())
				ss << "_abs_";
		}

		if (entry_subsampling_size != 1)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "samples " << entry_subsampling_size;
		}

		if (alpha != -std::numeric_limits<float>::max())
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "alpha " << alpha;
		}

		res.push_back(ss.str());

		return res;
	}

	float average_subsampling_layer::get_effective_alpha(
		const layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific) const
	{
		float res;
		if (alpha == -std::numeric_limits<float>::max())
		{
			unsigned int subsampling_elem_count = get_fm_subsampling_size(input_configuration_specific.feature_map_count, output_configuration_specific.feature_map_count) * entry_subsampling_size;
			for(unsigned int i = 0; i < static_cast<int>(output_configuration_specific.dimension_sizes.size()); ++i)
				subsampling_elem_count *= get_subsampling_size(i, input_configuration_specific.dimension_sizes[i], output_configuration_specific.dimension_sizes[i]);
			res = 1.0F / static_cast<float>(subsampling_elem_count);
		}
		else
			res = alpha;

		return res;
	}
}
