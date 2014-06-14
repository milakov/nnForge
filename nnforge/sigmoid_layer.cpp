/*
 *  Copyright 2011-2014 Maxim Milakov
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
#include "layer_factory.h"

namespace nnforge
{
	// {705604F3-D7E7-498D-A72C-318BF0ABFDE3}
	const boost::uuids::uuid sigmoid_layer::layer_guid =
		{ 0x70, 0x56, 0x04, 0xf3
		, 0xd7, 0xe7
		, 0x49, 0x8d
		, 0xa7, 0x2c
		, 0x31, 0x8b, 0xf0, 0xab, 0xfd, 0xe3 };

	sigmoid_layer::sigmoid_layer()
	{
	}

	const boost::uuids::uuid& sigmoid_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr sigmoid_layer::clone() const
	{
		return layer_smart_ptr(new sigmoid_layer(*this));
	}

	float sigmoid_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 4);
	}

	float sigmoid_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 2);
	}

	float sigmoid_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 3);
	}
}
