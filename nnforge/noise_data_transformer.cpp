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

#include "noise_data_transformer.h"

#include "neural_network_exception.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	noise_data_transformer::noise_data_transformer(unsigned int max_noise)
	{
		generator = rnd::get_random_generator();

		max_noise_distribution = std::tr1::uniform_int<int>(-static_cast<int>(max_noise), static_cast<int>(max_noise));
	}

	noise_data_transformer::~noise_data_transformer()
	{
	}

	void noise_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("noise_data_transformer is implemented for data stored as bytes only");

		unsigned char * dt = static_cast<unsigned char *>(data_transformed);
		unsigned int elem_count = original_config.get_neuron_count();

		for(unsigned char * data_it = dt; data_it != (dt + elem_count); data_it++)
		{
			int shift = max_noise_distribution(generator);
			*data_it = static_cast<unsigned char>(std::min<int>(std::max<int>((shift + static_cast<int>(*data_it)), 0), 255));
		}
	}
}
