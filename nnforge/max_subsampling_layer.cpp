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

#include "max_subsampling_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	const std::string max_subsampling_layer::layer_type_name = "MaxSubsampling";

	max_subsampling_layer::max_subsampling_layer(
		const std::vector<unsigned int>& subsampling_sizes,
		bool tiling)
		: subsampling_sizes(subsampling_sizes)
		, tiling(tiling)
	{
		check();
	}

	void max_subsampling_layer::check()
	{
		if (subsampling_sizes.size() == 0)
			throw neural_network_exception("subsampling sizes for max subsampling layer may not be empty");

		for(unsigned int i = 0; i < subsampling_sizes.size(); i++)
		{
			if (subsampling_sizes[i] == 0)
				throw neural_network_exception("window dimension for max subsampling layer may not be zero");
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

	layer_configuration max_subsampling_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if ((input_configuration_list[0].dimension_count >= 0) && (input_configuration_list[0].dimension_count != static_cast<int>(subsampling_sizes.size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration_list[0].dimension_count).str());

		return layer_configuration(input_configuration_list[0].feature_map_count, static_cast<int>(subsampling_sizes.size()));
	}

	layer_configuration_specific max_subsampling_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_dimension_count() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % subsampling_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific_list[0].feature_map_count);

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
				if (input_configuration_specific_list[0].dimension_sizes[i] < subsampling_sizes[i])
					throw neural_network_exception((boost::format("Input configuration size (%1%) of dimension (%2%) is smaller than subsampling size (%3%)") % input_configuration_specific_list[0].dimension_sizes[i] % i % subsampling_sizes[i]).str());

				res.dimension_sizes.push_back(input_configuration_specific_list[0].dimension_sizes[i] / subsampling_sizes[i]);
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

		input_configuration_specific = layer_configuration_specific(output_configuration_specific.feature_map_count);

		if (tiling)
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
				input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] * subsampling_sizes[i] + (subsampling_sizes[i] - 1));
		}
		else
		{
			for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
				input_configuration_specific.dimension_sizes.push_back(output_configuration_specific.dimension_sizes[i] * subsampling_sizes[i]);
		}

		return true;
	}

	std::vector<std::pair<unsigned int, unsigned int> > max_subsampling_layer::get_input_rectangle_borders(
		const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		unsigned int input_layer_id) const
	{
		if (output_rectangle_borders.size() != subsampling_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % subsampling_sizes.size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res;

		for(unsigned int i = 0; i < subsampling_sizes.size(); ++i)
			res.push_back(std::make_pair(output_rectangle_borders[i].first * subsampling_sizes[i], output_rectangle_borders[i].second * subsampling_sizes[i] + (subsampling_sizes[i] - 1)));

		return res;
	}

	void max_subsampling_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::MaxSubsamplingParam * param = layer_proto_typed->mutable_max_subsampling_param();
		for(int i = 0; i < subsampling_sizes.size(); ++i)
		{
			protobuf::MaxSubsamplingParam_MaxSubsamplingDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_subsampling_size(subsampling_sizes[i]);
		}

		if (tiling)
			param->set_tiling(true);
	}

	void max_subsampling_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_max_subsampling_param())
			throw neural_network_exception((boost::format("No max_subsampling_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		subsampling_sizes.resize(layer_proto_typed->max_subsampling_param().dimension_param_size());
		for(int i = 0; i < layer_proto_typed->max_subsampling_param().dimension_param_size(); ++i)
		{
			subsampling_sizes[i] = layer_proto_typed->max_subsampling_param().dimension_param(i).subsampling_size();
		}

		tiling = layer_proto_typed->max_subsampling_param().tiling();

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
				unsigned int per_item_flops = 1;
				std::for_each(subsampling_sizes.begin(), subsampling_sizes.end(), per_item_flops *= boost::lambda::_1);
				per_item_flops -= 1;
				if (tiling)
					neuron_count *= get_tiling_factor();
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			return 0.0F;
		default:
			return 0.0F;
		}
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
}
