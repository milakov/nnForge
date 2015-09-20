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
	structured_data_bunch_stream_reader::structured_data_bunch_stream_reader(const std::map<std::string, structured_data_reader::ptr>& data_reader_map)
		: data_reader_map(data_reader_map)
	{
		entry_count = -1;
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
		{
			int new_entry_count = it->second->get_entry_count();
			if (new_entry_count >= 0)
			{
				if (entry_count < 0)
					entry_count = new_entry_count;
				else if (entry_count != new_entry_count)
					throw std::runtime_error((boost::format("Entry count mismatch: %1% and %2%") % entry_count % new_entry_count).str());
			}
		}
	}

	structured_data_bunch_stream_reader::~structured_data_bunch_stream_reader()
	{
	}

	std::map<std::string, layer_configuration_specific> structured_data_bunch_stream_reader::get_config_map() const
	{
		std::map<std::string, layer_configuration_specific> res;
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
			res.insert(std::make_pair(it->first, it->second->get_configuration()));
		return res;
	}

	bool structured_data_bunch_stream_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		if ((entry_count >= 0) && (entry_id >= static_cast<unsigned int>(entry_count)))
			return false;

		bool res = true;
		for(std::map<std::string, float *>::const_iterator it = data_map.begin(); it != data_map.end(); ++it)
		{
			std::map<std::string, structured_data_reader::ptr>::const_iterator reader_it = data_reader_map.find(it->first);
			if (reader_it == data_reader_map.end())
				throw neural_network_exception((boost::format("structured_data_bunch_stream_reader is requested to read %1% data, while it doesn't have it") % it->first).str());
			res &= reader_it->second->read(entry_id, it->second);
		}
		return res;
	}

	void structured_data_bunch_stream_reader::next_epoch() const
	{
		for(std::map<std::string, structured_data_reader::ptr>::const_iterator it = data_reader_map.begin(); it != data_reader_map.end(); ++it)
			it->second->next_epoch();
	}

	int structured_data_bunch_stream_reader::get_entry_count() const
	{
		return entry_count;
	}
}
