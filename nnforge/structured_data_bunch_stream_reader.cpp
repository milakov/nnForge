/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "neural_network_exception.h"

namespace nnforge
{
	structured_data_bunch_stream_reader::structured_data_bunch_stream_reader(
		const std::map<std::string, structured_data_reader::ptr>& data_reader_map,
		unsigned int multiple_epoch_count)
		: data_reader_map(data_reader_map)
		, entry_count_list(multiple_epoch_count)
		, base_entry_count_list(multiple_epoch_count)
		, current_epoch(0)
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

		structured_data_bunch_stream_reader::ptr res(new structured_data_bunch_stream_reader(narrow_data_reader_map, static_cast<unsigned int>(entry_count_list.size())));
		res->current_epoch = current_epoch;
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
			throw neural_network_exception(invalid_config_message);

		current_epoch = epoch_id % entry_count_list.size();
	}

	bool structured_data_bunch_stream_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		if (!invalid_config_message.empty())
			throw neural_network_exception(invalid_config_message);

		if ((entry_count_list[current_epoch] >= 0) && (entry_id >= static_cast<unsigned int>(entry_count_list[current_epoch])))
			return false;

		unsigned int global_entry_id = entry_id + base_entry_count_list[current_epoch];
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
		return entry_count_list[current_epoch];
	}
}
