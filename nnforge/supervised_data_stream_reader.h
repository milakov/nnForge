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
#include "supervised_data_stream_schema.h"
#include "neural_network_exception.h"
#include "neuron_data_type.h"
#include "nn_types.h"

#include <vector>
#include <istream>
#include <ostream>

namespace nnforge
{
	class supervised_data_stream_reader : public supervised_data_reader
	{
	public:
		// The constructor modifies input_stream to throw exceptions in case of failure
		supervised_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream);

		virtual ~supervised_data_stream_reader();

		virtual void reset();

		virtual bool read(
			void * input_neurons,
			float * output_neurons);

		virtual bool raw_read(std::vector<unsigned char>& all_elems);

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

		virtual unsigned int get_entry_count() const
		{
			return entry_count;
		}

		virtual void rewind(unsigned int entry_id);

	protected:
		bool entry_available();

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		unsigned int input_neuron_count;
		unsigned int output_neuron_count;
		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		neuron_data_type::input_type type_code;
		unsigned int entry_count;

		unsigned int entry_read_count;
		std::istream::pos_type reset_pos;

	private:
		supervised_data_stream_reader(const supervised_data_stream_reader&);
		supervised_data_stream_reader& operator =(const supervised_data_stream_reader&);
	};

	typedef nnforge_shared_ptr<supervised_data_stream_reader> supervised_data_stream_reader_smart_ptr;
}
