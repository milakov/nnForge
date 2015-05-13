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

#include "sigmoid_layer.h"

#include "proto/nnforge.pb.h"

namespace nnforge
{
	// {705604F3-D7E7-498D-A72C-318BF0ABFDE3}
	const boost::uuids::uuid sigmoid_layer::layer_guid =
		{ 0x70, 0x56, 0x04, 0xf3
		, 0xd7, 0xe7
		, 0x49, 0x8d
		, 0xa7, 0x2c
		, 0x31, 0x8b, 0xf0, 0xab, 0xfd, 0xe3 };

	const std::string sigmoid_layer::layer_type_name = "Sigmoid";

	sigmoid_layer::sigmoid_layer(const std::vector<unsigned int>& affected_feature_map_id_list)
		: affected_feature_map_id_list(affected_feature_map_id_list)
	{
	}

	const boost::uuids::uuid& sigmoid_layer::get_uuid() const
	{
		return layer_guid;
	}

	const std::string& sigmoid_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer_smart_ptr sigmoid_layer::clone() const
	{
		return layer_smart_ptr(new sigmoid_layer(*this));
	}

	float sigmoid_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		if (affected_feature_map_id_list.empty())
			return static_cast<float>(input_configuration_specific.get_neuron_count() * 4);
		else
			return static_cast<float>(input_configuration_specific.get_neuron_count_per_feature_map() * affected_feature_map_id_list.size() * 4);
	}

	float sigmoid_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		if (affected_feature_map_id_list.empty())
			return static_cast<float>(input_configuration_specific.get_neuron_count() * 2);
		else
			return static_cast<float>(input_configuration_specific.get_neuron_count_per_feature_map() * affected_feature_map_id_list.size() * 2);
	}

	void sigmoid_layer::write_proto(void * layer_proto) const
	{
		if (!affected_feature_map_id_list.empty())
		{
			protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
			protobuf::SigmoidParam * param = layer_proto_typed->mutable_sigmoid_param();

			for(std::vector<unsigned int>::const_iterator it = affected_feature_map_id_list.begin(); it != affected_feature_map_id_list.end(); ++it)
				param->add_feature_map_affected(*it);
		}
	}

	void sigmoid_layer::read_proto(const void * layer_proto)
	{
		affected_feature_map_id_list.clear();
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (layer_proto_typed->has_sigmoid_param())
		{
			for(int i = 0; i < layer_proto_typed->sigmoid_param().feature_map_affected_size(); ++i)
				affected_feature_map_id_list.push_back(layer_proto_typed->sigmoid_param().feature_map_affected(i));
		}
	}
}
