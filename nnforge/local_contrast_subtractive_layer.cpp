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

#include "local_contrast_subtractive_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <math.h>

namespace nnforge
{
	const std::string local_contrast_subtractive_layer::layer_type_name = "LCS";

	const float local_contrast_subtractive_layer::c = 2.0F;

	local_contrast_subtractive_layer::local_contrast_subtractive_layer(
		const std::vector<unsigned int>& window_sizes,
		const std::vector<unsigned int>& feature_maps_affected,
		unsigned int feature_map_count)
		: window_sizes(window_sizes)
		, feature_maps_affected(feature_maps_affected)
		, feature_map_count(feature_map_count)
	{
		if (this->feature_maps_affected.empty())
			for(unsigned int i = 0; i < feature_map_count; i++)
				this->feature_maps_affected.push_back(i);

		check_and_update();
	}

	void local_contrast_subtractive_layer::check_and_update()
	{
		if (window_sizes.size() == 0)
			throw neural_network_exception("window sizes for local contrast subtractive layer may not be empty");

		for(unsigned int i = 0; i < window_sizes.size(); i++)
		{
			if (window_sizes[i] == 0)
				throw neural_network_exception("window dimension for local contrast subtractive layer may not be zero");
		}

		std::sort(this->feature_maps_affected.begin(), this->feature_maps_affected.end());
		for(unsigned int i = 0; i < feature_map_count; i++)
		{
			if (!std::binary_search(this->feature_maps_affected.begin(), this->feature_maps_affected.end(), i))
			{
				feature_maps_unaffected.push_back(i);
			}
		}

		setup_window_weights_list();
	}

	std::string local_contrast_subtractive_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr local_contrast_subtractive_layer::clone() const
	{
		return layer::ptr(new local_contrast_subtractive_layer(*this));
	}

	layer_configuration_specific local_contrast_subtractive_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count != feature_map_count)
			throw neural_network_exception((boost::format("Feature map count in layer (%1%) and input configuration (%2%) don't match") % feature_map_count % input_configuration_specific_list[0].feature_map_count).str());

		if (input_configuration_specific_list[0].get_dimension_count() != window_sizes.size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % window_sizes.size() % input_configuration_specific_list[0].get_dimension_count()).str());

		for(unsigned int i = 0; i < window_sizes.size(); i++)
		{
			if (input_configuration_specific_list[0].dimension_sizes[i] < window_sizes[i])
				throw neural_network_exception((boost::format("Input configuration dimension size (%1%) for local contrast subtractive layer is smaller than window size (%2%") % input_configuration_specific_list[0].dimension_sizes[i] % window_sizes[i]).str());
		}

		for(std::vector<unsigned int>::const_iterator it = feature_maps_affected.begin(); it != feature_maps_affected.end(); ++it)
		{
			if (*it >= input_configuration_specific_list[0].feature_map_count)
				throw neural_network_exception((boost::format("ID of feature map layer (%1%) is greater or equal than feature map count of input configuration (%2%)") % *it % input_configuration_specific_list[0].feature_map_count).str());
		}

		return input_configuration_specific_list[0];
	}

	void local_contrast_subtractive_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::LCSParam * param = layer_proto_typed->mutable_lcs_param();

		param->set_feature_map_count(feature_map_count);
		for(int i = 0; i < feature_maps_affected.size(); ++i)
		{
			param->add_feature_map_affected(feature_maps_affected[i]);
		}
		for(int i = 0; i < window_sizes.size(); ++i)
		{
			protobuf::LCSParam_LCSDimensionParam * dim_param = param->add_dimension_param();
			dim_param->set_kernel_size(window_sizes[i]);
		}
	}

	void local_contrast_subtractive_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_lcs_param())
			throw neural_network_exception((boost::format("No lcs_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		feature_map_count = layer_proto_typed->lcs_param().feature_map_count();
		feature_maps_affected.resize(layer_proto_typed->lcs_param().feature_map_affected_size());
		for(int i = 0; i < layer_proto_typed->lcs_param().feature_map_affected_size(); ++i)
		{
			feature_maps_affected[i] = layer_proto_typed->lcs_param().feature_map_affected(i);
		}

		window_sizes.resize(layer_proto_typed->lcs_param().dimension_param_size());
		for(int i = 0; i < layer_proto_typed->lcs_param().dimension_param_size(); ++i)
		{
			window_sizes[i] = layer_proto_typed->lcs_param().dimension_param(i).kernel_size();
		}

		check_and_update();
	}

	float local_contrast_subtractive_layer::get_std_dev(unsigned int dimension_id) const
	{
		unsigned int window_dimension_size = window_sizes[dimension_id];

		if (window_dimension_size <= 1)
			return -1.0F;

		unsigned int m = (window_dimension_size - 1) >> 1;

		return static_cast<float>(m) / c;
	}

	float local_contrast_subtractive_layer::get_gaussian_value(
		int offset,
		unsigned int dimension_id) const
	{
		float std_dev = get_std_dev(dimension_id);

		if (std_dev < 0.0F)
			return 1.0F;

		return expf(static_cast<float>(offset * offset) / (-2.0F * std_dev * std_dev)) / (std_dev * sqrtf(2.0F * 3.141593F));
	}

	void local_contrast_subtractive_layer::setup_window_weights_list()
	{
		window_weights_list.resize(window_sizes.size());

		for(unsigned int dimension_id = 0; dimension_id < window_sizes.size(); ++dimension_id)
		{
			std::vector<float>& current_window_sizes = window_weights_list[dimension_id];
			unsigned int window_dimension_size = window_sizes[dimension_id];
			current_window_sizes.resize((window_dimension_size + 1) >> 1);

			if (current_window_sizes.size() > 0)
			{
				float sum = 0.0F;
				for(unsigned int offset = 0; offset < current_window_sizes.size(); ++offset)
				{
					float val = get_gaussian_value(offset, dimension_id);
					sum += val;
					if (offset > 0)
						sum += val;
					current_window_sizes[offset] = val;
				}
				float mult = 1.0F / sum;
				for(unsigned int offset = 0; offset < current_window_sizes.size(); ++offset)
				{
					current_window_sizes[offset] *= mult;
				}
			}
		}
	}

	float local_contrast_subtractive_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int neuron_count = static_cast<unsigned int>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * feature_maps_affected.size());
				unsigned int per_item_flops = 1;
				std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops += ((boost::lambda::_1 >> 1) * 3 + 1) );
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int neuron_count = static_cast<unsigned int>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * feature_maps_affected.size());
				unsigned int per_item_flops = 1;
				std::for_each(window_sizes.begin(), window_sizes.end(), per_item_flops += ((boost::lambda::_1 >> 1) * 3 + 1) );
				return static_cast<float>(neuron_count) * static_cast<float>(per_item_flops);
			}
		default:
			return 0.0F;
		}
	}
}
