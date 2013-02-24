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

#include "supervised_data_stream_schema.h"
#include "layer_configuration_specific.h"

#include <memory>
#include <vector>
#include <ostream>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class supervised_data_stream_writer_base
	{
	protected:
		// The stream should be created with std::ios_base::binary flag
		supervised_data_stream_writer_base(
			std::tr1::shared_ptr<std::ostream> output_stream,
			const layer_configuration_specific& input_configuration,
			const layer_configuration_specific& output_configuration,
			unsigned int type_code);

		virtual ~supervised_data_stream_writer_base();

		void write_output(const float * output_elems);

		std::tr1::shared_ptr<std::ostream> out_stream;
		unsigned int input_neuron_count;
		unsigned int output_neuron_count;

	private:
		supervised_data_stream_writer_base();

		std::ostream::pos_type entry_count_pos;
		unsigned int entry_count;
	};

	template <typename input_data_type, unsigned int data_type_code> class supervised_data_stream_writer : public supervised_data_stream_writer_base
	{
	public:
		// The constructor modifies output_stream to throw exceptions in case of failure
		supervised_data_stream_writer(
			std::tr1::shared_ptr<std::ostream> output_stream,
			const layer_configuration_specific& input_configuration,
			const layer_configuration_specific& output_configuration)
			: supervised_data_stream_writer_base(output_stream, input_configuration, output_configuration, data_type_code)
		{
		}

		virtual ~supervised_data_stream_writer()
		{
		}

		void write(
			const input_data_type * input_neurons,
			const float * output_neurons)
		{
			out_stream->write(reinterpret_cast<const char*>(input_neurons), sizeof(*input_neurons) * input_neuron_count);

			supervised_data_stream_writer_base::write_output(output_neurons);
		}

	private:
		supervised_data_stream_writer(const supervised_data_stream_writer&);
		supervised_data_stream_writer& operator =(const supervised_data_stream_writer&);
	};

	typedef supervised_data_stream_writer<unsigned char, supervised_data_stream_schema::type_char> supervised_data_stream_writer_byte;
	typedef supervised_data_stream_writer<float, supervised_data_stream_schema::type_float> supervised_data_stream_writer_float;
}
