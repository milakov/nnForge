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

#include "reshape_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <sstream>
#include <boost/format.hpp>

namespace nnforge
{
	const std::string reshape_layer::layer_type_name = "Reshape";

	reshape_layer::reshape_layer()
		: entry_factor(1)
		, feature_map_factor(1)
		, collapse_to_dimensions(-1)
	{
	}

	std::string reshape_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr reshape_layer::clone() const
	{
		return layer::ptr(new reshape_layer(*this));
	}

	layer_configuration reshape_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if (input_configuration_list[0].dimension_count != dimension_factor_list.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % dimension_factor_list.size() % input_configuration_list[0].dimension_count).str());

		layer_configuration res;

		res.dimension_count = input_configuration_list[0].dimension_count;
		if ((res.dimension_count >= 0) && (collapse_to_dimensions >= 0))
			res.dimension_count = std::min(input_configuration_list[0].dimension_count, collapse_to_dimensions);

		res.feature_map_count = input_configuration_list[0].feature_map_count;
		if (res.feature_map_count >= 0)
			res.feature_map_count = tiling_factor(res.feature_map_count) * feature_map_factor;

		return res;
	}

	layer_configuration_specific reshape_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_dimension_count() != dimension_factor_list.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % dimension_factor_list.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific_list[0]);

		res.feature_map_count = tiling_factor(res.feature_map_count) * feature_map_factor;
		for(unsigned int i = 0; i < dimension_factor_list.size(); ++i)
			res.dimension_sizes[i] = tiling_factor(res.dimension_sizes[i]) * dimension_factor_list[i];

		if (collapse_to_dimensions >= 0)
		{
			for(unsigned int i = collapse_to_dimensions; i < res.dimension_sizes.size(); ++i)
			{
				if (res.dimension_sizes[i] != 1)
					throw neural_network_exception((boost::format("Cannot collapse dimension %1% of size %2%") % i % res.dimension_sizes[i]).str());
			}
			res.dimension_sizes.resize(collapse_to_dimensions);
		}

		return res;
	}

	bool reshape_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		unsigned int target_dimension_count = static_cast<unsigned int>(dimension_factor_list.size());
		if (collapse_to_dimensions >= 0)
			target_dimension_count = std::min(target_dimension_count, static_cast<unsigned int>(collapse_to_dimensions));

		if (output_configuration_specific.get_dimension_count() != target_dimension_count)
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % target_dimension_count % output_configuration_specific.get_dimension_count()).str());

		input_configuration_specific = layer_configuration_specific(output_configuration_specific);
		if (dimension_factor_list.size() > input_configuration_specific.dimension_sizes.size())
			input_configuration_specific.dimension_sizes.resize(dimension_factor_list.size(), 1);

		input_configuration_specific.feature_map_count = tiling_factor(input_configuration_specific.feature_map_count) * feature_map_factor.get_inverse();
		for(unsigned int i = 0; i < dimension_factor_list.size(); ++i)
			input_configuration_specific.dimension_sizes[i] = tiling_factor(input_configuration_specific.dimension_sizes[i]) * dimension_factor_list[i].get_inverse();

		return true;
	}

	void reshape_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::ReshapeParam * param = layer_proto_typed->mutable_reshape_param();

		if (entry_factor != tiling_factor(1))
		{
			protobuf::ReshapeParam_EntryParam * entry_param = param->mutable_entry_param();
			if (entry_factor > tiling_factor(1))
				entry_param->set_upsampling_size(entry_factor);
			else
				entry_param->set_downsampling_size(entry_factor.get_inverse());
		}

		if (feature_map_factor != tiling_factor(1))
		{
			protobuf::ReshapeParam_FeatureMapParam * feature_map_param = param->mutable_feature_map_param();
			if (feature_map_factor > tiling_factor(1))
				feature_map_param->set_upsampling_size(feature_map_factor);
			else
				feature_map_param->set_downsampling_size(feature_map_factor.get_inverse());
		}

		for(int i = 0; i < dimension_factor_list.size(); ++i)
		{
			const tiling_factor& dimension_factor = dimension_factor_list[i];
			protobuf::ReshapeParam_DimensionParam * dimension_param = param->add_dimension_param();
			if (dimension_factor > tiling_factor(1))
				dimension_param->set_upsampling_size(dimension_factor);
			else
				dimension_param->set_downsampling_size(dimension_factor.get_inverse());
		}

		if (collapse_to_dimensions >= 0)
			param->set_collapse_to_dimensions(collapse_to_dimensions);
	}

	void reshape_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_reshape_param())
			throw neural_network_exception((boost::format("No reshape_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::ReshapeParam& param = layer_proto_typed->reshape_param();

		if (param.has_entry_param())
		{
			if (param.entry_param().has_downsampling_size())
			{
				if (param.entry_param().has_upsampling_size())
					throw neural_network_exception("entry_param cannot have both upsamping and downsampling sizes");
				else
					entry_factor = tiling_factor(param.entry_param().downsampling_size()).get_inverse();
			}
			else
			{
				if (param.entry_param().has_upsampling_size())
					entry_factor = tiling_factor(param.entry_param().upsampling_size());
				else
					entry_factor = 1;
			}
		}

		if (param.has_feature_map_param())
		{
			if (param.feature_map_param().has_downsampling_size())
			{
				if (param.feature_map_param().has_upsampling_size())
					throw neural_network_exception("feature_map_param cannot have both upsamping and downsampling sizes");
				else
					feature_map_factor = tiling_factor(param.feature_map_param().downsampling_size()).get_inverse();
			}
			else
			{
				if (param.feature_map_param().has_upsampling_size())
					feature_map_factor = tiling_factor(param.feature_map_param().upsampling_size());
				else
					feature_map_factor = 1;
			}
		}

		dimension_factor_list.resize(param.dimension_param_size());
		for(int i = 0; i < param.dimension_param_size(); ++i)
		{
			const protobuf::ReshapeParam_DimensionParam& dimension_param = param.dimension_param(i);

			if (dimension_param.has_downsampling_size())
			{
				if (dimension_param.has_upsampling_size())
					throw neural_network_exception("dimension_param cannot have both upsamping and downsampling sizes");
				else
					dimension_factor_list[i] = tiling_factor(dimension_param.downsampling_size()).get_inverse();
			}
			else
			{
				if (dimension_param.has_upsampling_size())
					dimension_factor_list[i] = tiling_factor(dimension_param.upsampling_size());
				else
					dimension_factor_list[i] = 1;
			}
		}

		if (param.has_collapse_to_dimensions())
			collapse_to_dimensions = param.collapse_to_dimensions();
		else
			collapse_to_dimensions = -1;

		check();
	}

	void reshape_layer::check()
	{
		tiling_factor cumulative_factor = entry_factor * feature_map_factor;
		for(std::vector<tiling_factor>::const_reverse_iterator it = dimension_factor_list.rbegin(); it != dimension_factor_list.rend(); ++it)
			cumulative_factor *= *it;
		if (cumulative_factor != tiling_factor(1))
			throw neural_network_exception((boost::format("Invalid cumulative tiling factor for reshape_layer: %1%") % cumulative_factor.str()).str());
	}

	float reshape_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return 0.0F;
		case layer_action::backward_data:
			return 0.0F;
		default:
			return 0.0F;
		}
	}

	tiling_factor reshape_layer::get_tiling_factor() const
	{
		return entry_factor;
	}

	std::vector<std::string> reshape_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		ss << "upsampling ";
		for(int i = 0; i < dimension_factor_list.size(); ++i)
		{
			if (i != 0)
				ss << "x";
			ss << dimension_factor_list[i].str();
		}

		ss << ", fm " << feature_map_factor.str();
		ss << ", samples " << entry_factor.str();

		res.push_back(ss.str());

		return res;
	}
}
