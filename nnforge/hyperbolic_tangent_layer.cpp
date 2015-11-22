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

#include "hyperbolic_tangent_layer.h"

#include "proto/nnforge.pb.h"

namespace nnforge
{
	const std::string hyperbolic_tangent_layer::layer_type_name = "TanH";

	hyperbolic_tangent_layer::hyperbolic_tangent_layer(
		float scale,
		float steepness)
		: scale(scale)
		, steepness(steepness)
	{
	}

	std::string hyperbolic_tangent_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr hyperbolic_tangent_layer::clone() const
	{
		return layer::ptr(new hyperbolic_tangent_layer(*this));
	}

	float hyperbolic_tangent_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count() * 6);
		case layer_action::backward_data:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count() * 5);
		default:
			return 0.0F;
		}
	}

	void hyperbolic_tangent_layer::write_proto(void * layer_proto) const
	{
		if ((scale != 1.0F) || (steepness != 1.0F))
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::TanHParam * param = layer_proto_typed->mutable_tanh_param();

			if (scale != 1.0F)
				param->set_scale(scale);
			if (steepness != 1.0F)
				param->set_steepness(steepness);
		}
	}

	void hyperbolic_tangent_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_tanh_param())
		{
			scale = 1.0F;
			steepness = 1.0F;
		}
		else
		{
			scale = layer_proto_typed->tanh_param().scale();
			steepness = layer_proto_typed->tanh_param().steepness();
		}
	}
}
