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

#include "hyperbolic_tangent_layer.h"
#include "layer_factory.h"

namespace nnforge
{
	// {1EF8AEEA-0E72-40A9-BA37-E82B1870EFF3}
	const boost::uuids::uuid hyperbolic_tangent_layer::layer_guid =
		{ 0x1e, 0xf8, 0xae, 0xea
		, 0x0e, 0x72
		, 0x40, 0xa9
		, 0xba, 0x37
		, 0xe8, 0x2b, 0x18, 0x70, 0xef, 0xf3 };

	const float hyperbolic_tangent_layer::steepness = 0.666666F;
	const float hyperbolic_tangent_layer::major_multiplier = 1.7159F;

	hyperbolic_tangent_layer::hyperbolic_tangent_layer()
	{
	}

	const boost::uuids::uuid& hyperbolic_tangent_layer::get_uuid() const
	{
		return layer_guid;
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

	float hyperbolic_tangent_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count() * 6);
	}
}
