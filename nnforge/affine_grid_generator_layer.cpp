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

#include "affine_grid_generator_layer.h"

#include "neural_network_exception.h"
#include "nn_types.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <numeric>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string affine_grid_generator_layer::layer_type_name = "AffineGridGenerator";

	affine_grid_generator_layer::affine_grid_generator_layer(
		const std::vector<unsigned int>& output_sizes,
		bool adjust_for_zero_init)
		: output_sizes(output_sizes)
		, adjust_for_zero_init(adjust_for_zero_init)
	{
		check();
	}

	void affine_grid_generator_layer::check()
	{
		if (output_sizes.size() != 2)
			throw neural_network_exception((boost::format("affine_grid_generator_layer is able to generate with 2D grid only, while %1% output sizes specified") % output_sizes.size()).str());

		for(unsigned int i = 0; i < output_sizes.size(); i++)
			if (output_sizes[i] == 0)
				throw neural_network_exception("output size for affine_grid_generator_layer may not be zero");
	}

	std::string affine_grid_generator_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr affine_grid_generator_layer::clone() const
	{
		return layer::ptr(new affine_grid_generator_layer(*this));
	}

	layer_configuration_specific affine_grid_generator_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].get_neuron_count_per_feature_map() != 1)
			throw neural_network_exception((boost::format("Wrong number of input neuron count per feature map for affine_grid_generator_layer %1%, while %2% expected") % input_configuration_specific_list[0].get_neuron_count_per_feature_map() % 1).str());

		if (input_configuration_specific_list[0].feature_map_count != 6)
			throw neural_network_exception((boost::format("Wrong number of input feature maps for affine_grid_generator_layer %1%, while %2% expected") % input_configuration_specific_list[0].feature_map_count % 6).str());

		return layer_configuration_specific(static_cast<unsigned int>(output_sizes.size()), output_sizes);
	}

	bool affine_grid_generator_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}

	void affine_grid_generator_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::AffineGridGeneratorParam * param = layer_proto_typed->mutable_affine_grid_generator_param();

		if (!adjust_for_zero_init)
			param->set_adjust_for_zero_init(false);

		for(int i = 0; i < output_sizes.size(); ++i)
		{
			protobuf::AffineGridGeneratorParam_AffineGridGeneratorDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_output_size(output_sizes[i]);
		}
	}

	void affine_grid_generator_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_affine_grid_generator_param())
			throw neural_network_exception((boost::format("No affine_grid_generator_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		const nnforge::protobuf::AffineGridGeneratorParam& param = layer_proto_typed->affine_grid_generator_param();

		adjust_for_zero_init = param.adjust_for_zero_init();

		output_sizes.resize(param.dimension_param_size());
		for(int i = 0; i < param.dimension_param_size(); ++i)
		{
			output_sizes[i] = param.dimension_param(i).output_size();
		}

		check();
	}

	float affine_grid_generator_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = get_output_layer_configuration_specific(input_configuration_specific_list).get_neuron_count();
				unsigned int per_item_flops = 4;
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				// FIXME!!!
			}
		case layer_action::backward_weights:
		default:
			return 0.0F;
		}
	}

	std::vector<std::string> affine_grid_generator_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;

		for(int i = 0; i < output_sizes.size(); ++i)
		{
			if (i != 0)
				ss << "x";
			ss << output_sizes[i];
		}

		if (adjust_for_zero_init)
			ss << ", adjust for zero init";

		res.push_back(ss.str());

		return res;
	}
}
