/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "soft_rectified_linear_layer.h"
#include "layer_factory.h"

namespace nnforge
{
	// {47D2C929-32BC-4651-8760-A8AAF8170970}
	const boost::uuids::uuid soft_rectified_linear_layer::layer_guid =
		{ 0x47, 0xd2, 0xc9, 0x29
		, 0x32, 0xbc
		, 0x46, 0x51
		, 0x87, 0x60
		, 0xa8, 0xaa, 0xf8, 0x17, 0x9, 0x70 };

	soft_rectified_linear_layer::soft_rectified_linear_layer()
	{
	}

	const boost::uuids::uuid& soft_rectified_linear_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr soft_rectified_linear_layer::clone() const
	{
		return layer_smart_ptr(new soft_rectified_linear_layer(*this));
	}

	layer_configuration soft_rectified_linear_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		return layer_configuration(input_configuration);
	}

	layer_configuration_specific soft_rectified_linear_layer::get_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		return layer_configuration_specific(input_configuration_specific);
	}

	float soft_rectified_linear_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 3);
	}

	float soft_rectified_linear_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 4);
	}

	float soft_rectified_linear_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 5);
	}
}
