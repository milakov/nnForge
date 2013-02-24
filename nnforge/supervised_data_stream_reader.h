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
#include "supervised_data_stream_writer.h"
#include "rnd.h"
#include "neural_network_exception.h"

#include <memory>
#include <vector>
#include <istream>
#include <ostream>
#include <random>

namespace nnforge
{
	class supervised_data_stream_reader_base
	{
	public:
		void rewind(unsigned int entry_id);

	protected:
		supervised_data_stream_reader_base(
			std::tr1::shared_ptr<std::istream> input_stream,
			size_t input_elem_size,
			unsigned int type_code);

		~supervised_data_stream_reader_base();

		void read_output(float * output_neurons);

		bool entry_available();

		void reset();

		std::tr1::shared_ptr<std::istream> in_stream;

		unsigned int input_neuron_count;
		unsigned int output_neuron_count;
		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		unsigned int entry_count;

	private:
		unsigned int entry_read_count;

		std::istream::pos_type reset_pos;

		size_t input_elem_size;

		supervised_data_stream_reader_base();
	};

	class randomized_classifier_keeper
	{
	public:
		randomized_classifier_keeper();

		bool is_empty();

		float get_ratio();

		void push(unsigned int entry_id);

		unsigned int peek_random(random_generator& rnd);

	protected:
		std::vector<unsigned int> entry_id_list;
		unsigned int pushed_count;
		float remaining_ratio;

		void update_ratio();
	};

	template <typename input_data_type, unsigned int data_type_code> class supervised_data_stream_reader : public supervised_data_reader<input_data_type>, public supervised_data_stream_reader_base
	{
	public:
		// The constructor modifies output_stream to throw exceptions in case of failure
		supervised_data_stream_reader(std::tr1::shared_ptr<std::istream> input_stream)
			: supervised_data_stream_reader_base(input_stream, sizeof(input_data_type), data_type_code)
		{
		}

		virtual ~supervised_data_stream_reader()
		{
		}

		virtual bool read(
			input_data_type * input_neurons,
			float * output_neurons)
		{
			if (!entry_available())
				return false;

			if (input_neurons)
				in_stream->read(reinterpret_cast<char*>(input_neurons), sizeof(*input_neurons) * input_neuron_count);
			else
				in_stream->seekg(sizeof(*input_neurons) * input_neuron_count, std::ios_base::cur);

			supervised_data_stream_reader_base::read_output(output_neurons);

			return true;
		}

		void write_randomized(std::tr1::shared_ptr<std::ostream> output_stream)
		{
			supervised_data_stream_writer<input_data_type, data_type_code> sw(
				output_stream,
				input_configuration,
				output_configuration);

			if (entry_count == 0)
				return;

			random_generator rnd = rnd::get_random_generator();

			std::vector<unsigned int> entry_to_write_list(entry_count);
			for(unsigned int i = 0; i < entry_count; ++i)
			{
				entry_to_write_list[i] = i;
			}

			std::vector<input_data_type> in(input_neuron_count);
			std::vector<float> out(output_neuron_count);

			for(unsigned int entry_to_write_count = entry_count; entry_to_write_count > 0; --entry_to_write_count)
			{
				std::tr1::uniform_int<unsigned int> dist(0, entry_to_write_count - 1);

				unsigned int index = dist(rnd);
				unsigned int entry_id = entry_to_write_list[index];

				rewind(entry_id);
				read(&(*in.begin()), &(*out.begin()));
				sw.write(&(*in.begin()), &(*out.begin()));

				unsigned int leftover_entry_id = entry_to_write_list[entry_to_write_count - 1];
				entry_to_write_list[index] = leftover_entry_id;
			}
		}

		void write_randomized_classifier(std::tr1::shared_ptr<std::ostream> output_stream)
		{
			supervised_data_stream_writer<input_data_type, data_type_code> sw(
				output_stream,
				input_configuration,
				output_configuration);

			if (entry_count == 0)
				return;

			random_generator rnd = rnd::get_random_generator();

			std::vector<randomized_classifier_keeper> class_buckets_entry_id_lists;
			fill_class_buckets_entry_id_lists(class_buckets_entry_id_lists);

			std::vector<input_data_type> in(input_neuron_count);
			std::vector<float> out(output_neuron_count);

			for(unsigned int entry_to_write_count = entry_count; entry_to_write_count > 0; --entry_to_write_count)
			{
				std::vector<randomized_classifier_keeper>::iterator bucket_it = class_buckets_entry_id_lists.begin();
				float best_ratio = 0.0F;
				for(std::vector<randomized_classifier_keeper>::iterator it = class_buckets_entry_id_lists.begin(); it != class_buckets_entry_id_lists.end(); ++it)
				{
					float new_ratio = it->get_ratio();
					if (new_ratio > best_ratio)
					{
						bucket_it = it;
						best_ratio = new_ratio;
					}
				}

				if (bucket_it->is_empty())
					throw neural_network_exception("Unexpected error in write_randomized_classifier: No elements left");

				unsigned int entry_id = bucket_it->peek_random(rnd);

				rewind(entry_id);
				read(&(*in.begin()), &(*out.begin()));
				sw.write(&(*in.begin()), &(*out.begin()));
			}
		}

		virtual void reset()
		{
			supervised_data_stream_reader_base::reset();
		}

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual layer_configuration_specific get_output_configuration() const
		{
			return output_configuration;
		}

		virtual unsigned int get_entry_count() const
		{
			return entry_count;
		}

	protected:
		void fill_class_buckets_entry_id_lists(std::vector<randomized_classifier_keeper>& class_buckets_entry_id_lists)
		{
			class_buckets_entry_id_lists.resize(output_neuron_count + 1);

			std::vector<float> output(output_neuron_count);

			unsigned int entry_id = 0;
			while (read(0, &(*output.begin())))
			{
				float min_value = *std::min_element(output.begin(), output.end());
				std::vector<float>::iterator max_elem = std::max_element(output.begin(), output.end());

				if ((min_value < *max_elem) || (*max_elem > 0.0F))
					class_buckets_entry_id_lists[max_elem - output.begin()].push(entry_id);
				else
					class_buckets_entry_id_lists[output.size()].push(entry_id);

				++entry_id;
			}
		}

	private:
		supervised_data_stream_reader(const supervised_data_stream_reader&);
		supervised_data_stream_reader& operator =(const supervised_data_stream_reader&);
	};

	typedef supervised_data_stream_reader<unsigned char, supervised_data_stream_schema::type_char> supervised_data_stream_reader_byte;
	typedef supervised_data_stream_reader<float, supervised_data_stream_schema::type_float> supervised_data_stream_reader_float;
}
