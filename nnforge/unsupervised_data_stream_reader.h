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
#include "rnd.h"
#include "neural_network_exception.h"

#include <memory>
#include <istream>

namespace nnforge
{
	class unsupervised_data_stream_reader_base
	{
	protected:
		unsupervised_data_stream_reader_base(
			std::tr1::shared_ptr<std::istream> input_stream,
			size_t input_elem_size,
			unsigned int type_code);
		
		~unsupervised_data_stream_reader_base();

		bool entry_available();

		void reset();

		std::tr1::shared_ptr<std::istream> in_stream;

		unsigned int input_neuron_count;
		layer_configuration_specific input_configuration;
		unsigned int entry_count;

	protected:
		void notify_read();

	private:
		unsigned int entry_read_count;

		std::istream::pos_type reset_pos;

		size_t input_elem_size;

		unsupervised_data_stream_reader_base();
	};

	template <typename input_data_type, unsigned int data_type_code> class unsupervised_data_stream_reader : public unsupervised_data_reader<input_data_type>, public unsupervised_data_stream_reader_base
	{
	public:
		// The constructor modifies output_stream to throw exceptions in case of failure
		unsupervised_data_stream_reader(std::tr1::shared_ptr<std::istream> input_stream)
			: unsupervised_data_stream_reader_base(input_stream, sizeof(input_data_type), data_type_code)
		{
		}

		virtual ~unsupervised_data_stream_reader()
		{
		}

		virtual bool read(input_data_type * input_neurons)
		{
			if (!entry_available())
				return false;

			if (input_neurons)
				in_stream->read(reinterpret_cast<char*>(input_neurons), sizeof(*input_neurons) * input_neuron_count);
			else
				in_stream->seekg(sizeof(*input_neurons) * input_neuron_count, std::ios_base::cur);

			unsupervised_data_stream_reader_base::notify_read();

			return true;
		}

		virtual void reset()
		{
			unsupervised_data_stream_reader_base::reset();
		}

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual unsigned int get_entry_count() const
		{
			return entry_count;
		}

	private:
		unsupervised_data_stream_reader(const unsupervised_data_stream_reader&);
		unsupervised_data_stream_reader& operator =(const unsupervised_data_stream_reader&);
	};

	typedef unsupervised_data_stream_reader<unsigned char, unsupervised_data_stream_schema::type_char> unsupervised_data_stream_reader_byte;
	typedef unsupervised_data_stream_reader<float, unsupervised_data_stream_schema::type_float> unsupervised_data_stream_reader_float;
}
