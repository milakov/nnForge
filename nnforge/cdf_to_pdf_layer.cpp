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

#include "cdf_to_pdf_layer.h"

#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <boost/format.hpp>
#include <sstream>

namespace nnforge
{
	const std::string cdf_to_pdf_layer::layer_type_name = "CDF2PDF";

	cdf_to_pdf_layer::cdf_to_pdf_layer(
		unsigned int feature_map_segment_length,
		float clamp_min,
		float clamp_max)
		: feature_map_segment_length(feature_map_segment_length)
		, clamp_min(clamp_min)
		, clamp_max(clamp_max)
	{
		check();
	}

	void cdf_to_pdf_layer::check()
	{
		if (feature_map_segment_length < 2)
			throw neural_network_exception("feature_map_segment_length for cdf2pdf layer should be at least 2");

		if (clamp_min >= clamp_max)
			throw neural_network_exception("clamp_min should be smaller than clamp_max for cdf2pdf layer");
	}

	std::string cdf_to_pdf_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer::ptr cdf_to_pdf_layer::clone() const
	{
		return layer::ptr(new cdf_to_pdf_layer(*this));
	}

	layer_configuration_specific cdf_to_pdf_layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		if (input_configuration_specific_list[0].feature_map_count % feature_map_segment_length != 0)
			throw neural_network_exception((boost::format("Feature map count in input configuration (%1%) is not evenly divisible by feature map segment length (%2%)") % input_configuration_specific_list[0].feature_map_count % feature_map_segment_length).str());

		return input_configuration_specific_list[0];
	}

	bool cdf_to_pdf_layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		if (output_configuration_specific.feature_map_count % feature_map_segment_length != 0)
			throw neural_network_exception((boost::format("Feature map count in output configuration (%1%) is not evenly divisible by feature map segment length (%2%)") % output_configuration_specific.feature_map_count % feature_map_segment_length).str());

		input_configuration_specific = output_configuration_specific;

		return true;
	}

	void cdf_to_pdf_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<nnforge::protobuf::Layer *>(layer_proto);
		nnforge::protobuf::CDF2PDFParam * param = layer_proto_typed->mutable_cdf_to_pdf_param();

		param->mutable_feature_map_param()->set_segment_length(feature_map_segment_length);

		if (clamp_min != -std::numeric_limits<float>::max())
			param->set_clamp_min(clamp_min);

		if (clamp_max != std::numeric_limits<float>::max())
			param->set_clamp_max(clamp_max);
	}

	void cdf_to_pdf_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const nnforge::protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_cdf_to_pdf_param())
			throw neural_network_exception((boost::format("No cdf_to_pdf_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		const protobuf::CDF2PDFParam& param = layer_proto_typed->cdf_to_pdf_param();

		if (!param.has_feature_map_param())
			throw neural_network_exception((boost::format("No feature_map_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());
		feature_map_segment_length = param.feature_map_param().segment_length();

		clamp_min = param.has_clamp_min() ? param.clamp_min() : -std::numeric_limits<float>::max();

		clamp_max = param.has_clamp_max() ? param.clamp_max() : std::numeric_limits<float>::max();

		check();
	}

	float cdf_to_pdf_layer::get_flops_per_entry(
		const std::vector<layer_configuration_specific>& input_configuration_specific_list,
		const layer_action& action) const
	{
		switch (action.get_action_type())
		{
		case layer_action::forward:
			{
				unsigned int segments_per_feature_map = (input_configuration_specific_list[0].feature_map_count / feature_map_segment_length);
				unsigned int per_fm_flops = segments_per_feature_map * (feature_map_segment_length - 1);
				return static_cast<float>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * per_fm_flops);
			}
		case layer_action::backward_data:
			{
				unsigned int segments_per_feature_map = (input_configuration_specific_list[0].feature_map_count / feature_map_segment_length);
				unsigned int per_fm_flops = segments_per_feature_map * (feature_map_segment_length - 1);
				return static_cast<float>(input_configuration_specific_list[0].get_neuron_count_per_feature_map() * per_fm_flops);
			}
		default:
			return 0.0F;
		}
	}

	std::vector<std::string> cdf_to_pdf_layer::get_parameter_strings() const
	{
		std::vector<std::string> res;

		std::stringstream ss;
		ss << "length fm " << feature_map_segment_length;
		if ((clamp_min != -std::numeric_limits<float>::max()) || (clamp_max != std::numeric_limits<float>::max()))
		{
			ss << ", clamp [";
			if (clamp_min != -std::numeric_limits<float>::max())
				ss << clamp_min;
			ss << ";";
			if (clamp_max != std::numeric_limits<float>::max())
				ss << clamp_max;
			ss << "]";
		}

		res.push_back(ss.str());

		return res;
	}
}
