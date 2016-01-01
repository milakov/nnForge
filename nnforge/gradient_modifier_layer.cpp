/*
 *  Copyright 2011-6 Maxim Milakov
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

#include "gradient_modifier_layer.h"

#include "proto/nnforge.pb.h"

#include <boost/format.hpp>

namespace nnforge
{
	const std::string gradient_modifier_layer::layer_type_name = "GradientModifier";

	gradient_modifier_layer::gradient_modifier_layer(float scale)
		: scale(scale)
	{
	}

	std::string gradient_modifier_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr gradient_modifier_layer::clone() const
	{
		return layer::ptr(new gradient_modifier_layer(*this));
	}

	float gradient_modifier_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return 0.0F;
		case layer_action::backward_data:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count());
		default:
			return 0.0F;
		}
	}

	void gradient_modifier_layer::write_proto(void * layer_proto) const
	{
		if (scale != 1.0F)
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::GradientModifierParam * param = layer_proto_typed->mutable_gradient_modifier_param();

			if (scale != 1.0F)
				param->set_scale(scale);
		}
	}

	void gradient_modifier_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_gradient_modifier_param())
		{
			scale = 1.0F;
		}
		else
		{
			scale = layer_proto_typed->gradient_modifier_param().scale();
		}
	}

	std::vector<std::string> gradient_modifier_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		res.push_back((boost::format("gradient scale %|1$.3f|") % scale).str());

		return res;
	}
}
