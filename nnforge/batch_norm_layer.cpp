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

#include "batch_norm_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string batch_norm_layer::layer_type_name = "BatchNorm";
	const float batch_norm_layer::default_batch_normalization_epsilon = 2.0e-5F;

	batch_norm_layer::batch_norm_layer(
		unsigned int feature_map_count,
		float epsilon)
		: feature_map_count(feature_map_count)
		, epsilon(epsilon)
	{
		check();
	}

	void batch_norm_layer::check()
	{
		if (feature_map_count == 0)
			throw neural_network_exception("Feature map count should be > 0 for batch norm layer");
	}

	std::string batch_norm_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr batch_norm_layer::clone() const
	{
		return layer::ptr(new batch_norm_layer(*this));
	}

	layer_configuration batch_norm_layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		if (input_configuration_list[0].feature_map_count >= 0)
		{
			if (input_configuration_list[0].feature_map_count != feature_map_count)
				throw neural_network_exception((boost::format("Feature map count in layer (%1%) is not equal to feature map count (%2%) in batch_norm_layer") % input_configuration_list[0].feature_map_count % feature_map_count).str());
		}

		return input_configuration_list[0];
	}

	float batch_norm_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		unsigned int neuron_count_per_feature_map = input_configuration_specific_list[0].get_neuron_count_per_feature_map();
		unsigned int feature_map_count = input_configuration_specific_list[0].feature_map_count;
		switch (action.get_action_type())
		{
		case layer_action::forward:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count() * 5);
		case layer_action::backward_data_and_weights:
			return static_cast<float>(input_configuration_specific_list[0].get_neuron_count() * 13);
		default:
			return 0.0F;
		}
	}

	void batch_norm_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::BatchNormParam * param = layer_proto_typed->mutable_batch_norm_param();

		param->set_feature_map_count(feature_map_count);

		if (epsilon != default_batch_normalization_epsilon)
			param->set_epsilon(epsilon);
	}

	void batch_norm_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_batch_norm_param())
			throw neural_network_exception((boost::format("No batch_norm_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		feature_map_count = layer_proto_typed->batch_norm_param().feature_map_count();

		epsilon = layer_proto_typed->batch_norm_param().has_epsilon() ? layer_proto_typed->batch_norm_param().epsilon() : default_batch_normalization_epsilon;

		check();
	}

	data_config batch_norm_layer::get_data_config() const
	{
		data_config res;

		res.push_back(feature_map_count);
		res.push_back(feature_map_count);
		res.push_back(feature_map_count);
		res.push_back(feature_map_count);

		return res;
	}

	void batch_norm_layer::randomize_data(
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
		std::fill((*data)[0].begin(), (*data)[0].end(), 1.0F);
		std::fill((*data)[1].begin(), (*data)[1].end(), 0.0F);
		std::fill((*data)[2].begin(), (*data)[2].end(), 0.0F);
		std::fill((*data)[3].begin(), (*data)[3].end(), 1.0F);
	}

	layer_data_configuration_list batch_norm_layer::get_layer_data_configuration_list() const
	{
		layer_data_configuration_list res;

		res.push_back(layer_data_configuration(feature_map_count, 1, std::vector<unsigned int>()));
		res.push_back(layer_data_configuration(feature_map_count, 1, std::vector<unsigned int>()));
		res.push_back(layer_data_configuration(feature_map_count, 1, std::vector<unsigned int>()));
		res.push_back(layer_data_configuration(feature_map_count, 1, std::vector<unsigned int>()));

		return res;
	}

	std::set<unsigned int> batch_norm_layer::get_weight_decay_part_id_set() const
	{
		return std::set<unsigned int>();
	}

	std::vector<std::string> batch_norm_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		ss << "fm " << feature_map_count;

		if (epsilon != default_batch_normalization_epsilon)
			ss << ", epsilon = " << epsilon;

		res.push_back(ss.str());

		return res;
	}

	bool batch_norm_layer::has_fused_backward_data_and_weights() const
	{
		return true;
	}
}
