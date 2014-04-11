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

#pragma once

#include "supervised_data_reader.h"
#include "rnd.h"
#include "neuron_data_type.h"

#include <memory>
#include <vector>

namespace nnforge
{
	class supervised_data_mem_reader : public supervised_data_reader
	{
	public:
		supervised_data_mem_reader(
			const layer_configuration_specific& input_configuration,
			const layer_configuration_specific& output_configuration,
			const std::vector<nnforge_shared_ptr<const std::vector<unsigned char> > >& input_data_list,
			const std::vector<nnforge_shared_ptr<const std::vector<float> > >& output_data_list);

		supervised_data_mem_reader(
			const layer_configuration_specific& input_configuration,
			const layer_configuration_specific& output_configuration,
			const std::vector<nnforge_shared_ptr<const std::vector<float> > >& input_data_list,
			const std::vector<nnforge_shared_ptr<const std::vector<float> > >& output_data_list);

		virtual ~supervised_data_mem_reader();

		virtual void reset()
		{
			entry_read_count = 0;
		}

		virtual bool read(
			void * input_neurons,
			float * output_neurons);

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual layer_configuration_specific get_output_configuration() const
		{
			return output_configuration;
		}

		virtual neuron_data_type::input_type get_input_type() const
		{
			return type_code;
		}

	protected:
		virtual unsigned int get_actual_entry_count() const
		{
			return entry_count;
		}

		bool entry_available()
		{
			return (entry_read_count < get_entry_count());
		}

	protected:
		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		neuron_data_type::input_type type_code;
		std::vector<nnforge_shared_ptr<const std::vector<unsigned char> > > input_data_list_byte;
		std::vector<nnforge_shared_ptr<const std::vector<float> > > input_data_list_float;
		std::vector<nnforge_shared_ptr<const std::vector<float> > > output_data_list;

		unsigned int entry_read_count;
		unsigned int entry_count;
		unsigned int input_neuron_count;
		unsigned int output_neuron_count;

	private:
		supervised_data_mem_reader(const supervised_data_mem_reader&);
		supervised_data_mem_reader& operator =(const supervised_data_mem_reader&);
	};

	typedef nnforge_shared_ptr<supervised_data_mem_reader> supervised_data_mem_reader_smart_ptr;
}
