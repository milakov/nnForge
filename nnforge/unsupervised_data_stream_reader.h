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

#include "unsupervised_data_reader.h"
#include "unsupervised_data_stream_schema.h"
#include "neural_network_exception.h"
#include "neuron_data_type.h"

#include <memory>
#include <vector>
#include <istream>
#include <ostream>

namespace nnforge
{
	class unsupervised_data_stream_reader : public unsupervised_data_reader
	{
	public:
		// The constructor modifies input_stream to throw exceptions in case of failure
		unsupervised_data_stream_reader(std::tr1::shared_ptr<std::istream> input_stream);

		virtual ~unsupervised_data_stream_reader();

		virtual void reset();

		virtual bool read(void * input_neurons);

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual unsigned int get_entry_count() const
		{
			return entry_count;
		}

		virtual neuron_data_type::input_type get_input_type() const
		{
			return type_code;
		}

	protected:
		bool entry_available();

	protected:
		std::tr1::shared_ptr<std::istream> in_stream;
		unsigned int input_neuron_count;
		layer_configuration_specific input_configuration;
		neuron_data_type::input_type type_code;
		unsigned int entry_count;

		unsigned int entry_read_count;
		std::istream::pos_type reset_pos;

	private:
		unsupervised_data_stream_reader(const unsupervised_data_stream_reader&);
		unsupervised_data_stream_reader& operator =(const unsupervised_data_stream_reader&);
	};

	typedef std::tr1::shared_ptr<unsupervised_data_stream_reader> unsupervised_data_stream_reader_smart_ptr;
}
