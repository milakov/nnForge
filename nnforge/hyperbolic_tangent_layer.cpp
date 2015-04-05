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
	// {1EF8AEEA-0E72-40A9-BA37-E82B1870EFF3}
	const boost::uuids::uuid hyperbolic_tangent_layer::layer_guid =
		{ 0x1e, 0xf8, 0xae, 0xea
		, 0x0e, 0x72
		, 0x40, 0xa9
		, 0xba, 0x37
		, 0xe8, 0x2b, 0x18, 0x70, 0xef, 0xf3 };

	const std::string hyperbolic_tangent_layer::layer_type_name = "TanH";

	hyperbolic_tangent_layer::hyperbolic_tangent_layer(
		float scale,
		float steepness)
		: scale(scale)
		, steepness(steepness)
	{
	}

	const boost::uuids::uuid& hyperbolic_tangent_layer::get_uuid() const
	{
		return layer_guid;
	}

	const std::string& hyperbolic_tangent_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer_smart_ptr hyperbolic_tangent_layer::clone() const
	{
		return layer_smart_ptr(new hyperbolic_tangent_layer(*this));
	}

	float hyperbolic_tangent_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 6);
	}

	float hyperbolic_tangent_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 5);
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

	void hyperbolic_tangent_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		scale = 1.7159F;
		steepness = 0.666666F;
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
