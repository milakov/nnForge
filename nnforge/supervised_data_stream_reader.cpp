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

#include "supervised_data_stream_reader.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	supervised_data_stream_reader::supervised_data_stream_reader(nnforge_shared_ptr<std::istream> input_stream)
		: in_stream(input_stream)
		, entry_read_count(0)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != supervised_data_stream_schema::supervised_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown supervised data GUID encountered in input stream: %1%") % guid_read).str());

		input_configuration.read(*in_stream);
		output_configuration.read(*in_stream);

		input_neuron_count = input_configuration.get_neuron_count();
		output_neuron_count = output_configuration.get_neuron_count();

		unsigned int type_code_read;
		in_stream->read(reinterpret_cast<char*>(&type_code_read), sizeof(type_code_read));
		type_code = static_cast<neuron_data_type::input_type>(type_code_read);

		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

		reset_pos = in_stream->tellg();
	}

	supervised_data_stream_reader::~supervised_data_stream_reader()
	{
	}

	void supervised_data_stream_reader::reset()
	{
		in_stream->seekg(reset_pos);

		entry_read_count = 0;
	}

	bool supervised_data_stream_reader::read(
		void * input_neurons,
		float * output_neurons)
	{
		if (!entry_available())
			return false;

		if (input_neurons)
			in_stream->read(reinterpret_cast<char*>(input_neurons), get_input_neuron_elem_size() * input_neuron_count);
		else
			in_stream->seekg(get_input_neuron_elem_size() * input_neuron_count, std::ios_base::cur);

		if (output_neurons)
			in_stream->read(reinterpret_cast<char*>(output_neurons), sizeof(*output_neurons) * output_neuron_count);
		else
			in_stream->seekg(sizeof(*output_neurons) * output_neuron_count, std::ios_base::cur);

		entry_read_count++;

		return true;
	}

	bool supervised_data_stream_reader::entry_available()
	{
		return (entry_read_count < get_entry_count());
	}

	void supervised_data_stream_reader::rewind(unsigned int entry_id)
	{
		in_stream->seekg(reset_pos);
		in_stream->seekg((std::istream::off_type)entry_id * (std::istream::off_type)((get_input_neuron_elem_size() * input_neuron_count) + (sizeof(float) * output_neuron_count)), std::ios::cur);

		entry_read_count = entry_id;
	}

	void supervised_data_stream_reader::write_randomized(nnforge_shared_ptr<std::ostream> output_stream)
	{
		supervised_data_stream_writer sw(
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

		std::vector<unsigned char> in(input_neuron_count * get_input_neuron_elem_size());
		std::vector<float> out(output_neuron_count);

		for(unsigned int entry_to_write_count = entry_count; entry_to_write_count > 0; --entry_to_write_count)
		{
			nnforge_uniform_int_distribution<unsigned int> dist(0, entry_to_write_count - 1);

			unsigned int index = dist(rnd);
			unsigned int entry_id = entry_to_write_list[index];

			rewind(entry_id);
			read(&(*in.begin()), &(*out.begin()));
			sw.write(type_code, (const void *)(&(*in.begin())), &(*out.begin()));

			unsigned int leftover_entry_id = entry_to_write_list[entry_to_write_count - 1];
			entry_to_write_list[index] = leftover_entry_id;
		}
	}

	void supervised_data_stream_reader::write_randomized_classifier(nnforge_shared_ptr<std::ostream> output_stream)
	{
		supervised_data_stream_writer sw(
			output_stream,
			input_configuration,
			output_configuration);

		if (entry_count == 0)
			return;

		random_generator rnd = rnd::get_random_generator();

		std::vector<randomized_classifier_keeper> class_buckets_entry_id_lists;
		fill_class_buckets_entry_id_lists(class_buckets_entry_id_lists);

		std::vector<unsigned char> in(input_neuron_count * get_input_neuron_elem_size());
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
			sw.write(type_code, (const void *)(&(*in.begin())), &(*out.begin()));
		}
	}

	void supervised_data_stream_reader::fill_class_buckets_entry_id_lists(std::vector<randomized_classifier_keeper>& class_buckets_entry_id_lists)
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

	randomized_classifier_keeper::randomized_classifier_keeper()
		: pushed_count(0)
		, remaining_ratio(0.0F)
	{
	}

	bool randomized_classifier_keeper::is_empty()
	{
		return entry_id_list.empty();
	}

	float randomized_classifier_keeper::get_ratio()
	{
		return remaining_ratio;
	}

	void randomized_classifier_keeper::push(unsigned int entry_id)
	{
		entry_id_list.push_back(entry_id);
		++pushed_count;

		update_ratio();
	}

	unsigned int randomized_classifier_keeper::peek_random(random_generator& rnd)
	{
		nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(entry_id_list.size()) - 1);

		unsigned int index = dist(rnd);
		unsigned int entry_id = entry_id_list[index];

		unsigned int leftover_entry_id = entry_id_list[entry_id_list.size() - 1];
		entry_id_list[index] = leftover_entry_id;

		entry_id_list.pop_back();

		update_ratio();

		return entry_id;
	}

	void randomized_classifier_keeper::update_ratio()
	{
		remaining_ratio = pushed_count > 0 ? static_cast<float>(entry_id_list.size()) / static_cast<float>(pushed_count) : 0.0F;
	}
}
