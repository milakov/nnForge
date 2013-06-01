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

#include "layer.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	layer::layer()
	{
	}

	layer::~layer()
	{
	}

	layer_data_smart_ptr layer::create_layer_data() const
	{
		layer_data_smart_ptr res(new layer_data());

		data_config dc = get_data_config();

		res->resize(dc.size());

		for(unsigned int i = 0; i < dc.size(); ++i)
		{
			(*res)[i].resize(dc[i]);
		}

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

	void layer::randomize_data(
		layer_data& data,
		random_generator& generator) const
	{
	}

	data_config layer::get_data_config() const
	{
		return data_config();
	}

	void layer::write(std::ostream& binary_stream_to_write_to) const
	{
	}

	void layer::read(std::istream& binary_stream_to_read_from)
	{
	}

	bool layer::is_empty_data() const
	{
		data_config dc = get_data_config();

		for(data_config::const_iterator it = dc.begin(); it != dc.end(); it++)
		{
			if (*it > 0)
				return false;
		}

		return true;
	}

	float layer::get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return 0.0F;
	}

	float layer::get_weights_update_flops_2nd(const layer_configuration_specific& input_configuration_specific) const
	{
		return 0.0F;
	}

	dropout_layer_config layer::get_dropout_layer_config(float dropout_rate) const
	{
		return dropout_layer_config();
	}

	layer_data_configuration_list layer::get_layer_data_configuration_list() const
	{
		return layer_data_configuration_list();
	}
}
