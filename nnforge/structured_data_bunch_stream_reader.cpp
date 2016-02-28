/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "structured_data_bunch_stream_reader.h"

#include <boost/format.hpp>
#include <limits>

#include "neural_network_exception.h"
#include "rnd.h"

namespace nnforge
{
	structured_data_bunch_stream_reader::structured_data_bunch_stream_reader(
		const std::map<std::string, structured_data_reader::ptr>& data_reader_map,
		unsigned int multiple_epoch_count,
		unsigned int shuffle_block_size)
		: data_reader_map(data_reader_map)
		, entry_count_list(multiple_epoch_count)
		, base_entry_count_list(multiple_epoch_count)
		, shuffle_block_size(shuffle_block_size)
		, current_epoch(0)
		, current_chunk(0)
		, current_big_epoch(0)
	{
		total_entry_count = -1;
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
		{
			int new_entry_count = it->second->get_entry_count();
			if (new_entry_count >= 0)
			{
				if (total_entry_count < 0)
					total_entry_count = new_entry_count;
				else if (total_entry_count != new_entry_count)
				{
					invalid_config_message = (boost::format("Entry count mismatch: %1% and %2%") % total_entry_count % new_entry_count).str();
					return;
				}
			}
		}

		if (shuffle_block_size > 0)
		{
			if (total_entry_count < 0)
			{
				invalid_config_message = "Shuffling specified for structured_data_bunch_stream_reader while entry count cannot be determined";
				return;
			}
			else
			{
				blocks_shuffled.resize(total_entry_count / shuffle_block_size);
				update_shuffle_list();
			}
		}

		if (multiple_epoch_count == 1)
		{
			entry_count_list[0] = total_entry_count;
		}
		else
		{
			if (total_entry_count < 0)
			{
				invalid_config_message = "Multiple epoch count specified for structured_data_bunch_stream_reader while entry count cannot be determined";
				return;
			}

			unsigned int epoch_min_size = total_entry_count / multiple_epoch_count;
			unsigned int plus1_epoch_count = total_entry_count % multiple_epoch_count;
			std::fill_n(entry_count_list.begin(), plus1_epoch_count, epoch_min_size + 1);
			std::fill_n(entry_count_list.begin() + plus1_epoch_count, multiple_epoch_count - plus1_epoch_count, epoch_min_size);
			for(unsigned int i = 1; i < static_cast<unsigned int>(base_entry_count_list.size()); ++i)
				base_entry_count_list[i] = base_entry_count_list[i - 1] + entry_count_list[i - 1];
		}
	}

	structured_data_bunch_stream_reader::~structured_data_bunch_stream_reader()
	{
	}

	structured_data_bunch_reader::ptr structured_data_bunch_stream_reader::get_narrow_reader(const std::set<std::string>& layer_names) const
	{
		std::map<std::string, structured_data_reader::ptr> narrow_data_reader_map;
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
			if (layer_names.find(it->first) != layer_names.end())
				narrow_data_reader_map.insert(*it);

		structured_data_bunch_stream_reader::ptr res(new structured_data_bunch_stream_reader(narrow_data_reader_map, static_cast<unsigned int>(entry_count_list.size()), shuffle_block_size));
		res->set_epoch(current_epoch);
		return res;
	}

	std::map<std::string, layer_configuration_specific> structured_data_bunch_stream_reader::get_config_map() const
	{
		std::map<std::string, layer_configuration_specific> res;
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
			res.insert(std::make_pair(it->first, it->second->get_configuration()));
		return res;
	}

	void structured_data_bunch_stream_reader::set_epoch(unsigned int epoch_id)
	{
		if (!invalid_config_message.empty())
			return;

		unsigned int new_current_big_epoch = epoch_id / static_cast<unsigned int>(entry_count_list.size());
		if (new_current_big_epoch != current_big_epoch)
		{
			current_big_epoch = new_current_big_epoch;
			if (shuffle_block_size > 0)
				update_shuffle_list();
		}

		current_chunk = epoch_id % entry_count_list.size();
		current_epoch = epoch_id;
	}

	bool structured_data_bunch_stream_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		if (!invalid_config_message.empty())
			throw neural_network_exception(invalid_config_message);

		if ((entry_count_list[current_chunk] >= 0) && (entry_id >= static_cast<unsigned int>(entry_count_list[current_chunk])))
			return false;

		unsigned int global_entry_id = entry_id + base_entry_count_list[current_chunk];
		if (shuffle_block_size > 0)
		{
			unsigned int shuffle_block_id = global_entry_id / shuffle_block_size;
			if (shuffle_block_id < static_cast<unsigned int>(blocks_shuffled.size()))
			{
				unsigned int internal_block_id = global_entry_id - shuffle_block_id * shuffle_block_size;
				global_entry_id = blocks_shuffled[shuffle_block_id] * shuffle_block_size + internal_block_id;
			}
		}

		bool res = true;
		for(std::map<std::string, float *>::const_iterator it = data_map.begin(); it != data_map.end(); ++it)
		{
			std::map<std::string, structured_data_reader::ptr>::const_iterator reader_it = data_reader_map.find(it->first);
			if (reader_it == data_reader_map.end())
				throw neural_network_exception((boost::format("structured_data_bunch_stream_reader is requested to read %1% data, while it doesn't have it") % it->first).str());
			res &= reader_it->second->read(global_entry_id, it->second);
		}
		return res;
	}

	int structured_data_bunch_stream_reader::get_entry_count() const
	{
		return entry_count_list[current_chunk];
	}

	void structured_data_bunch_stream_reader::update_shuffle_list()
	{
		random_generator gen = rnd::get_random_generator(current_big_epoch);
		unsigned int block_count = static_cast<unsigned int>(blocks_shuffled.size());
		for(unsigned int i = 0; i < block_count; ++i)
			blocks_shuffled[i] = i;
		for(int i = static_cast<int>(block_count) - 1; i > 0; --i)
		{
			nnforge_uniform_int_distribution<int> dist(0, i);
			int elem_id = dist(gen);
			std::swap(blocks_shuffled[elem_id], blocks_shuffled[i]);
		}
	}
}
