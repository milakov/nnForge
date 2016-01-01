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

#include "layer.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <sstream>

namespace nnforge
{
	layer::layer()
	{
	}

	layer::~layer()
	{
	}

	bool layer::is_empty_data() const
	{
		return get_data_config().empty();
	}

	bool layer::is_empty_data_custom() const
	{
		return get_data_custom_config().empty();
	}

	layer_data::ptr layer::create_layer_data() const
	{
		layer_data::ptr res(new layer_data());

		data_config dc = get_data_config();
		res->resize(dc.size());
		for(unsigned int i = 0; i < dc.size(); ++i)
			(*res)[i].resize(dc[i]);

		return res;
	}

	layer_data_custom::ptr layer::create_layer_data_custom() const
	{
		layer_data_custom::ptr res(new layer_data_custom());

		data_config dc = get_data_custom_config();
		res->resize(dc.size());
		for(unsigned int i = 0; i < dc.size(); ++i)
			(*res)[i].resize(dc[i], -1);

		return res;
	}

	void layer::check_layer_data_consistency(const layer_data& data) const
	{
		data_config dc = get_data_config();
		if (dc.size() != data.size())
			throw neural_network_exception((boost::format("data weight vector count %1% doesn't satisfy layer configuration %2%") % data.size() % dc.size()).str());
		for(unsigned int i = 0; i < dc.size(); ++i)
		{
			if (dc[i] != data[i].size())
				throw neural_network_exception((boost::format("data weight count %1% for vector %2% doesn't satisfy layer configuration %3%") % data[i].size() % i % dc[i]).str());
		}
	}

	void layer::check_layer_data_custom_consistency(const layer_data_custom& data_custom) const
	{
		data_custom_config dcc = get_data_custom_config();
		if (dcc.size() != data_custom.size())
			throw neural_network_exception((boost::format("custom data weight vector count %1% doesn't satisfy layer configuration %2%") % data_custom.size() % dcc.size()).str());
		for(unsigned int i = 0; i < dcc.size(); ++i)
		{
			if (dcc[i] != data_custom[i].size())
				throw neural_network_exception((boost::format("custom data weight count %1% for vector %2% doesn't satisfy layer configuration %3%") % data_custom[i].size() % i % dcc[i]).str());
		}
	}

	void layer::randomize_data(
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
	}

	void layer::randomize_orthogonal_data(
		layer_data::ptr data,
		layer_data_custom::ptr data_custom,
		random_generator& generator) const
	{
		randomize_data(
			data,
			data_custom,
			generator);
	}

	data_config layer::get_data_config() const
	{
		return data_config();
	}

	data_custom_config layer::get_data_custom_config() const
	{
		return data_custom_config();
	}

	void layer::read_proto(const void * layer_proto)
	{
	}

	void layer::write_proto(void * layer_proto) const
	{
	}

	layer_data_configuration_list layer::get_layer_data_configuration_list() const
	{
		return layer_data_configuration_list();
	}

	layer_configuration layer::get_layer_configuration(const std::vector<layer_configuration>& input_configuration_list) const
	{
		return input_configuration_list.front();
	}

	layer_configuration_specific layer::get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const
	{
		return input_configuration_specific_list.front();
	}

	bool layer::get_input_layer_configuration_specific(
		layer_configuration_specific& input_configuration_specific,
		const layer_configuration_specific& output_configuration_specific,
		unsigned int input_layer_id) const
	{
		input_configuration_specific = output_configuration_specific;

		return true;
	}

	std::vector<std::pair<unsigned int, unsigned int> > layer::get_input_rectangle_borders(
		const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
		unsigned int input_layer_id) const
	{
		return output_rectangle_borders;
	}

	std::set<unsigned int> layer::get_weight_decay_part_id_set() const
	{
		return std::set<unsigned int>();
	}

	std::vector<std::string> layer::get_parameter_strings() const
	{
		return std::vector<std::string>();
	}

	tiling_factor layer::get_tiling_factor() const
	{
		return 1;
	}

	std::string layer::get_string_for_average_data(
		const layer_configuration_specific& config,
		const std::vector<float>& data) const
	{
		std::stringstream s;
		s << instance_name << " = ";
		for(std::vector<float>::const_iterator it = data.begin(); it != data.end(); ++it)
		{
			if (it != data.begin())
				s << ", ";
			s << *it;
		}
		return s.str();
	}
}
