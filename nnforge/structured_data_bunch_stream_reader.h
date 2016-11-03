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

#pragma once

#include "structured_data_bunch_reader.h"
#include "structured_data_stream_reader.h"

#include <string>
#include <memory>

namespace nnforge
{
	class structured_data_bunch_stream_reader : public structured_data_bunch_reader
	{
	public:
		typedef std::shared_ptr<structured_data_bunch_stream_reader> ptr;

		structured_data_bunch_stream_reader(
			const std::map<std::string, structured_data_reader::ptr>& data_reader_map,
			unsigned int multiple_epoch_count,
			unsigned int shuffle_block_size);

		virtual ~structured_data_bunch_stream_reader() = default;

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const;

		// The method returns false in case the entry cannot be read
		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map);

		virtual int get_entry_count() const;

		virtual structured_data_bunch_reader::ptr get_narrow_reader(const std::set<std::string>& layer_names) const;

		virtual void set_epoch(unsigned int epoch_id);

	private:
		void update_shuffle_list();

	protected:
		std::map<std::string, structured_data_reader::ptr> data_reader_map;
		int total_entry_count;
		std::vector<int> entry_count_list;
		std::vector<int> base_entry_count_list;
		unsigned int shuffle_block_size;
		unsigned int current_epoch;
		unsigned int current_chunk;
		unsigned int current_big_epoch;
		std::string invalid_config_message;
		std::vector<unsigned int> blocks_shuffled;
	};
}
