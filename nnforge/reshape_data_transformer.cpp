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

#include "reshape_data_transformer.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	reshape_data_transformer::reshape_data_transformer(const layer_configuration_specific& config)
		: config(config)
	{
	}

	reshape_data_transformer::~reshape_data_transformer()
	{
	}

	void reshape_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.get_neuron_count() != config.get_neuron_count())
			throw neural_network_exception((boost::format("Neuron counts for reshape_data_transformer don't match: %1% and %2%") % original_config.get_neuron_count() % config.get_neuron_count()).str());
	}

	layer_configuration_specific reshape_data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		if (original_config.get_neuron_count() != config.get_neuron_count())
			throw neural_network_exception((boost::format("Neuron counts for reshape_data_transformer don't match: %1% and %2%") % original_config.get_neuron_count() % config.get_neuron_count()).str());

		return config;
	}

	bool reshape_data_transformer::is_in_place() const
	{
		return true;
	}

	bool reshape_data_transformer::is_deterministic() const
	{
		return true;
	}
}
