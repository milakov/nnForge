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
	// {E85380CA-965F-46A2-995B-797AF265F803}
	const boost::uuids::uuid dropout_layer::layer_guid =
		{ 0xe8, 0x53, 0x80, 0xca
		, 0x96, 0x5f
		, 0x46, 0xa2
		, 0x99, 0x5b
		, 0x79, 0x7a, 0xf2, 0x65, 0xf8, 0x3 };

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

	const boost::uuids::uuid& dropout_layer::get_uuid() const
	{
		return layer_guid;
	}

	const std::string& dropout_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer_smart_ptr dropout_layer::clone() const
	{
		return layer_smart_ptr(new dropout_layer(*this));
	}

	float dropout_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return 0.0F;
	}

	float dropout_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count());
	}

	void dropout_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(dropout_rate));
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

	void dropout_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&dropout_rate), sizeof(dropout_rate));
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
