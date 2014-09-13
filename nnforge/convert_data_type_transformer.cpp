/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "convert_data_type_transformer.h"

#include "neural_network_exception.h"

namespace nnforge
{
	convert_data_type_transformer::convert_data_type_transformer()
	{
	}

	convert_data_type_transformer::~convert_data_type_transformer()
	{
	}

	void convert_data_type_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("convert_data_type_transformer is converting from bytes only");

		unsigned int neron_count = original_config.get_neuron_count();

		const unsigned char * src_data = (const unsigned char *)data;
		float * dst_data = (float *)data_transformed;

		for(unsigned int i = 0; i < neron_count; ++i)
			dst_data[i] = static_cast<float>(src_data[i]) * (1.0F / 255.0F);
	}

 	bool convert_data_type_transformer::is_deterministic() const
	{
		return true;
	}

	bool convert_data_type_transformer::is_in_place() const
	{
		return false;
	}

	neuron_data_type::input_type convert_data_type_transformer::get_transformed_data_type(neuron_data_type::input_type original_data_type) const
	{
		return neuron_data_type::type_float;
	}
}
