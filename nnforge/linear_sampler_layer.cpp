/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "linear_sampler_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <numeric>
#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string linear_sampler_layer::layer_type_name = "LinearSampler";

	linear_sampler_layer::linear_sampler_layer()
	{
	}

	std::string linear_sampler_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr linear_sampler_layer::clone() const
	{
		return layer::ptr(new linear_sampler_layer(*this));
	}

	layer_configuration_specific linear_sampler_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list.size() != 2)
			throw neural_network_exception((boost::format("linear_sampler_layer: %1% input layers specified, while 2 are expected") % input_configuration_specific_list.size()).str());

		if (input_configuration_specific_list[0].dimension_sizes.size() != 2)
			throw neural_network_exception((boost::format("linear_sampler_layer is able to run in 2D only, while the grid has %1% dimensions") % input_configuration_specific_list[0].dimension_sizes.size()).str());

		if (input_configuration_specific_list[0].dimension_sizes.size() != input_configuration_specific_list[0].feature_map_count)
			throw neural_network_exception((boost::format("linear_sampler_layer: dimensions count mismatch for the grid: %1% and %2%") % input_configuration_specific_list[0].dimension_sizes.size() % input_configuration_specific_list[0].feature_map_count).str());

		if (input_configuration_specific_list[1].dimension_sizes.size() != input_configuration_specific_list[0].dimension_sizes.size())
			throw neural_network_exception((boost::format("linear_sampler_layer: sampled image has %1% dimensions, while grid has %2%") % input_configuration_specific_list[1].dimension_sizes.size() % input_configuration_specific_list[0].dimension_sizes.size()).str());

		return layer_configuration_specific(input_configuration_specific_list[1].feature_map_count, input_configuration_specific_list[0].dimension_sizes);
	}

	bool linear_sampler_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		return false;
	}

	float linear_sampler_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				layer_configuration_specific output_config = get_output_layer_configuration_specific(input_configuration_specific_list);
				unsigned int neuron_count_per_feature_map = output_config.get_neuron_count_per_feature_map();
				unsigned int per_feature_map_work = 7;
				unsigned int constant_work = 12;
				return static_cast<float>(neuron_count_per_feature_map) * static_cast<float>(constant_work + per_feature_map_work * output_config.feature_map_count);
			}
		case layer_action::backward_data:
			{
				layer_configuration_specific output_config = get_output_layer_configuration_specific(input_configuration_specific_list);
				unsigned int neuron_count_per_feature_map = output_config.get_neuron_count_per_feature_map();
				unsigned int per_feature_map_work = 8;
				unsigned int constant_work = 18;
				return static_cast<float>(neuron_count_per_feature_map) * static_cast<float>(constant_work + per_feature_map_work * output_config.feature_map_count);
			}
		case layer_action::backward_weights:
		default:
			return 0.0F;
		}
	}
}
