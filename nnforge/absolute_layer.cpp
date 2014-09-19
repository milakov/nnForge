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

#include "absolute_layer.h"

namespace nnforge
{
	// {55B9D9C0-3FB5-464B-9C06-49D0EC9F4249}
	const boost::uuids::uuid absolute_layer::layer_guid =
		{ 0x55, 0xb9, 0xd9, 0xc0
		, 0x3f, 0xb5
		, 0x46, 0x4b
		, 0x9c, 0x06
		, 0x49, 0xd0, 0xec, 0x9f, 0x42, 0x49 };

	absolute_layer::absolute_layer()
	{
	}

	const boost::uuids::uuid& absolute_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr absolute_layer::clone() const
	{
		return layer_smart_ptr(new absolute_layer(*this));
	}

	float absolute_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count());
	}

	float absolute_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count());
	}
}
