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

#include "max_subsampling_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string max_subsampling_layer::layer_type_name = "MaxSubsampling";

	max_subsampling_layer::max_subsampling_layer(
		const std::vector<unsigned int>& subsampling_sizes,
		unsigned int feature_map_subsampling_size,
		unsigned int entry_subsampling_size,
		bool is_min,
		bool tiling,
		const std::vector<bool>& round_ups,
		const std::vector<unsigned int>& strides)
		: subsampling_sizes(subsampling_sizes)
		, feature_map_subsampling_size(feature_map_subsampling_size)
		, entry_subsampling_size(entry_subsampling_size)
		, is_min(is_min)
		, tiling(tiling)
	{
		if ((round_ups.size() != 0) && (round_ups.size() != subsampling_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for round ups") % round_ups.size()).str());
		if ((strides.size() != 0) && (strides.size() != subsampling_sizes.size()))
			throw std::runtime_error((boost::format("Invalid dimension count %1% for strides") % strides.size()).str());

		if (round_ups.empty())
			this->round_ups.resize(subsampling_sizes.size(), false);
		else
			this->round_ups = round_ups;

		if (strides.empty())
			this->strides = subsampling_sizes;
		else
			this->strides = strides;

		check();
	}

	void max_subsampling_layer::check()
	{
		for(unsigned int i = 0; i < subsampling_sizes.size(); i++)
		{
			if (subsampling_sizes[i] == 0)
				throw neural_network_exception("window dimension for max subsampling layer may not be zero");
		}

		if (feature_map_subsampling_size == 0)
			throw neural_network_exception("feature map subsampling size for max subsampling layer may not be zero");

		if (entry_subsampling_size == 0)
			throw neural_network_exception("feature map subsampling size for max subsampling layer may not be zero");

		if (tiling && (entry_subsampling_size > 1))
			throw neural_network_exception("entry_subsampling_size cannot be set with tiling at the same time for max subsampling layer");

		if (tiling && (feature_map_subsampling_size > 1))
			throw neural_network_exception("feature_map_subsampling_size cannot be set with tiling at the same time for max subsampling layer");

		for(unsigned int i = 0; i < strides.size(); i++)
			if (strides[i] == 0)
				throw neural_network_exception((boost::format("stride dimension (%1%) is 0") % i).str());

		if (tiling)
		{
			for(unsigned int i = 0; i < round_ups.size(); i++)
			{
				if (round_ups[i])
					throw neural_network_exception("round up cannot be set with tiling at the same time for max subsampling layer");
			}
		}
	}

	std::string max_subsampling_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr max_subsampling_layer::clone() const
	{
		return layer::ptr(new max_subsampling_layer(*this));
	}

	layer_configuration_specific max_subsampling_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific_list[0].feature_map_count / feature_map_subsampling_size);

		if (tiling)
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
			{
				if (input_configuration_specific_list[0].dimension_sizes[i] < (subsampling_sizes[i] * 2 - 1))
					throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than subsampling size (%3%) * 2 - 1") % input_configuration_specific_list[0].dimension_sizes[i] % i % subsampling_sizes[i]).str());

				res.dimension_sizes.push_back((input_configuration_specific_list[0].dimension_sizes[i] - (subsampling_sizes[i] - 1))/ subsampling_sizes[i]);
			}
		}
		else
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
			{
				int new_size = (static_cast<int>(input_configuration_specific_list[0].dimension_sizes[i]) + static_cast<int>(strides[i]) - static_cast<int>(subsampling_sizes[i]) + (round_ups[i] ? static_cast<int>(subsampling_sizes[i]) - 1 : 0)) / static_cast<int>(strides[i]);

				if (new_size <= 0)
					throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) produces 0 output size") % input_configuration_specific_list[0].dimension_sizes[i] % i).str());

				res.dimension_sizes.push_back(static_cast<unsigned int>(new_size));
			}
		}

		return res;
	}

	bool max_subsampling_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % subsampling_sizes.size() % output_configuration_specific.get_dimension_count()).str());

		input_configuration_specific = layer_configuration_specific(output_configuration_specific.feature_map_count * feature_map_subsampling_size);

		if (tiling)
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
				input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] * subsampling_sizes[i] + (subsampling_sizes[i] - 1));
		}
		else
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
				input_configuration_specific.dimension_sizes.push_back((output_configuration_specific.dimension_sizes[i] - 1) * strides[i] + subsampling_sizes[i]);
		}

		return true;
	}

	void max_subsampling_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::MaxSubsamplingParam * param = layer_proto_typed->mutable_max_subsampling_param();
		for(int i = 0; i < subsampling_sizes.size(); ++i)
		{
			protobuf::MaxSubsamplingParam_MaxSubsamplingDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_subsampling_size(subsampling_sizes[i]);
			if (round_ups[i])
				dim_param->set_round_up(true);
			if (strides[i] != subsampling_sizes[i])
				dim_param->set_stride(strides[i]);
		}

		if (feature_map_subsampling_size != 1)
			param->mutable_feature_map_param()->set_subsampling_size(feature_map_subsampling_size);

		if (entry_subsampling_size != 1)
			param->mutable_entry_param()->set_subsampling_size(entry_subsampling_size);

		if (tiling)
			param->set_tiling(true);

		if (is_min)
			param->set_function(nnforge::protobuf::MaxSubsamplingParam_MaxFunction_MIN);
	}

	void max_subsampling_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_max_subsampling_param())
			throw neural_network_exception((boost::format("No max_subsampling_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::MaxSubsamplingParam& param = layer_proto_typed->max_subsampling_param();

		subsampling_sizes.resize(param.dimension_param_size());
		round_ups.resize(param.dimension_param_size());
		strides.resize(param.dimension_param_size());
		for(int i = 0; i < param.dimension_param_size(); ++i)
		{
			subsampling_sizes[i] = param.dimension_param(i).subsampling_size();
			round_ups[i] = param.dimension_param(i).round_up();
			strides[i] = param.dimension_param(i).has_stride() ? param.dimension_param(i).stride() : subsampling_sizes[i];
		}

		feature_map_subsampling_size = param.has_feature_map_param() ? param.feature_map_param().subsampling_size() : 1;

		entry_subsampling_size = param.has_entry_param() ? param.entry_param().subsampling_size() : 1;

		tiling = param.tiling();

		is_min = (param.function() == nnforge::protobuf::MaxSubsamplingParam_MaxFunction_MIN);

		check();
	}

	float max_subsampling_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = feature_map_subsampling_size * entry_subsampling_size;
				std::for_each(subsampling_sizes.begin(), subsampling_sizes.end(), [&per_item_flops] (unsigned int x) { per_item_flops *= x; });
				per_item_flops -= 1;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			return 0.0F;
		default:
			return 0.0F;
		}
	}

	tiling_factor max_subsampling_layer::get_tiling_factor() const
	{
		tiling_factor res;
		if (tiling)
		{
			std::vector<tiling_factor> tiling_factor_list = get_tiling_factor_list();

			res = 1;
			std::for_each(tiling_factor_list.begin(), tiling_factor_list.end(), [&res] (tiling_factor x) { res *= x; });
		}
		else
		{
			res = tiling_factor(entry_subsampling_size).get_inverse();
		}

		return res;
	}

	std::vector<tiling_factor> max_subsampling_layer::get_tiling_factor_list() const
	{
		std::vector<tiling_factor> res;

		if (tiling)
		{
			for(std::vector<unsigned int>::const_iterator it = subsampling_sizes.begin(); it != subsampling_sizes.end(); ++it)
				res.push_back(*it);
		}

		return res;
	}

	std::vector<std::string> max_subsampling_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		if (is_min)
			ss << "MIN";

		if (!subsampling_sizes.empty())
		{
			if (!ss.str().empty())
				ss << ", ";
			for(int i = 0; i < subsampling_sizes.size(); ++i)
			{
				if (i != 0)
					ss << "x";
				ss << subsampling_sizes[i];
				if (round_ups[i])
					ss << "_roundup_";
			}
		}

		bool nontrivial_stride = true;
		for(int i = 0; i < strides.size(); ++i)
		{
			if (strides[i] != subsampling_sizes[i])
			{
				nontrivial_stride = false;
				break;
			}
		}
		if (!nontrivial_stride)
		{
			ss << ", stride ";
			for(int i = 0; i < strides.size(); ++i)
			{
				if (i != 0)
					ss << "x";
				ss << strides[i];
			}
		}

		if (feature_map_subsampling_size != 1)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "fm " << feature_map_subsampling_size;
		}

		if (entry_subsampling_size != 1)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "samples " << entry_subsampling_size;
		}

		if (tiling)
		{
			if (!ss.str().empty())
				ss << ", ";
			ss << "tiling";
		}

		res.push_back(ss.str());

		return res;
	}
}
