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

#include "supervised_data_reader.h"

#include <vector>

namespace nnforge
{
	supervised_data_reader::supervised_data_reader()
	{
	}

	supervised_data_reader::~supervised_data_reader()
	{
	}

	size_t supervised_data_reader::get_input_neuron_elem_size() const
	{
		return neuron_data_type::get_input_size(get_input_type());
	}

	output_neuron_value_set_smart_ptr supervised_data_reader::get_output_neuron_value_set(unsigned int sample_count)
	{
		reset();

		unsigned int entry_count = get_entry_count();
		unsigned int output_neuron_count = get_output_configuration().get_neuron_count();

		output_neuron_value_set_smart_ptr res(new output_neuron_value_set(entry_count, output_neuron_count));

		for(std::vector<std::vector<float> >::iterator it = res->neuron_value_list.begin(); it != res->neuron_value_list.end(); it++)
		{
			std::vector<float>& output_neurons = *it;

			read(0, &(*output_neurons.begin()));
		}

		res->compact(sample_count);

		return res;
	}
}
