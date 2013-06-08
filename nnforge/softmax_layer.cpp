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

#include "softmax_layer.h"
#include "layer_factory.h"

namespace nnforge
{
	// {DB854721-6DD6-4B0F-B74E-DA63D0756EB5}
	const boost::uuids::uuid softmax_layer::layer_guid =
		{ 0xdb, 0x85, 0x47, 0x21
		, 0x6d, 0xd6
		, 0x4b, 0x0f
		, 0xb7, 0x4e
		, 0xda, 0x63, 0xd0, 0x75, 0x6e, 0xb5 };

	softmax_layer::softmax_layer()
	{
	}

	const boost::uuids::uuid& softmax_layer::get_uuid() const
	{
		return layer_guid;
	}

	layer_smart_ptr softmax_layer::clone() const
	{
		return layer_smart_ptr(new softmax_layer(*this));
	}

	layer_configuration softmax_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		return layer_configuration(input_configuration);
	}

	layer_configuration_specific softmax_layer::get_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		return layer_configuration_specific(input_configuration_specific);
	}

	float softmax_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count_per_feature_map() * (input_configuration_specific.feature_map_count * 3 - 1));
	}

	float softmax_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count_per_feature_map() * (input_configuration_specific.feature_map_count * 4 - 1));
	}

	float softmax_layer::get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(input_configuration_specific.get_neuron_count_per_feature_map() * (input_configuration_specific.feature_map_count * 9 - 1));
	}
}
