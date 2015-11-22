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

#include "dropout_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>

namespace nnforge
{
	const std::string dropout_layer::layer_type_name = "Dropout";

	dropout_layer::dropout_layer(float dropout_rate)
		: dropout_rate(dropout_rate)
	{
		check();
	}

	void dropout_layer::check()
	{
		if ((dropout_rate < 0.0F) || (dropout_rate >= 1.0F))
			throw neural_network_exception((boost::format("Error constructing dropout_layer: dropout_rate equals %1%, it should be in [0.0F,1.0F)") % dropout_rate).str());
	}

	std::string dropout_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr dropout_layer::clone() const
	{
		return layer::ptr(new dropout_layer(*this));
	}

	float dropout_layer::get_flops_per_entry(
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

	void dropout_layer::write_proto(void * layer_proto) const
	{
		if (dropout_rate != 0.5F)
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::DropoutParam * param = layer_proto_typed->mutable_dropout_param();
			param->set_dropout_rate(dropout_rate);
		}
	}

	void dropout_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_dropout_param())
		{
			dropout_rate = 0.5F;
		}
		else
		{
			dropout_rate = layer_proto_typed->dropout_param().dropout_rate();
		}

		check();
	}
}
