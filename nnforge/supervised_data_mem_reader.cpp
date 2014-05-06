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

#include "supervised_data_mem_reader.h"
#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <cstring>

namespace nnforge
{
	supervised_data_mem_reader::supervised_data_mem_reader(
		const layer_configuration_specific& input_configuration,
		const layer_configuration_specific& output_configuration,
		const std::vector<nnforge_shared_ptr<const std::vector<unsigned char> > >& input_data_list,
		const std::vector<nnforge_shared_ptr<const std::vector<float> > >& output_data_list)
		: input_configuration(input_configuration)
		, output_configuration(output_configuration)
		, input_data_list_byte(input_data_list)
		, output_data_list(output_data_list)
		, entry_read_count(0)
		, type_code(neuron_data_type::type_byte)
		, entry_count(static_cast<unsigned int>(input_data_list.size()))
		, input_neuron_count(input_configuration.get_neuron_count())
		, output_neuron_count(output_configuration.get_neuron_count())
	{
	}

	supervised_data_mem_reader::supervised_data_mem_reader(
		const layer_configuration_specific& input_configuration,
		const layer_configuration_specific& output_configuration,
		const std::vector<nnforge_shared_ptr<const std::vector<float> > >& input_data_list,
		const std::vector<nnforge_shared_ptr<const std::vector<float> > >& output_data_list)
		: input_configuration(input_configuration)
		, output_configuration(output_configuration)
		, input_data_list_float(input_data_list)
		, output_data_list(output_data_list)
		, entry_read_count(0)
		, type_code(neuron_data_type::type_float)
		, entry_count(static_cast<unsigned int>(input_data_list.size()))
		, input_neuron_count(input_configuration.get_neuron_count())
		, output_neuron_count(output_configuration.get_neuron_count())
	{
	}

	supervised_data_mem_reader::~supervised_data_mem_reader()
	{
	}

	bool supervised_data_mem_reader::read(
		void * input_neurons,
		float * output_neurons)
	{
		if (!entry_available())
			return false;

		if (input_neurons)
		{
			const void * input_src;
			switch (type_code)
			{
			case neuron_data_type::type_byte:
				input_src = &(*input_data_list_byte[entry_read_count]->begin());
				break;
			case neuron_data_type::type_float:
				input_src = &(*input_data_list_float[entry_read_count]->begin());
				break;
			}
			memcpy(input_neurons, input_src, input_neuron_count * neuron_data_type::get_input_size(type_code));
		}

		if (output_neurons)
		{
			const float * output_src = &(*output_data_list[entry_read_count]->begin());
			memcpy(output_neurons, output_src, input_neuron_count * sizeof(float));
		}

		entry_read_count++;

		return true;
	}
}
