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

#include "rectified_linear_layer.h"
#include "layer_factory.h"

namespace nnforge
{
	// {185FC3A0-5C2C-41B3-B483-5E9B0EFE877F}
	const boost::uuids::uuid rectified_linear_layer::layer_guid =
		{ 0x18, 0x5f, 0xc3, 0xa0
		, 0x5c, 0x2c
		, 0x41, 0xb3
		, 0xb4, 0x83
		, 0x5e, 0x9b, 0xe, 0xfe, 0x87, 0x7f };

	rectified_linear_layer::rectified_linear_layer()
	{
	}

	const boost::uuids::uuid& rectified_linear_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr rectified_linear_layer::clone() const
	{
		return layer_smart_ptr(new rectified_linear_layer(*this));
	}

	layer_configuration rectified_linear_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		return layer_configuration(input_configuration);
	}

	layer_configuration_specific rectified_linear_layer::get_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		return layer_configuration_specific(input_configuration_specific);
	}

	float rectified_linear_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count());
	}

	float rectified_linear_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(0.0F);
	}

	float rectified_linear_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(0.0F);
	}
}
