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

#pragma once

#include "structured_data_bunch_reader.h"
#include "structured_data_stream_reader.h"
#include "nn_types.h"

namespace nnforge
{
	class structured_data_bunch_stream_reader : public structured_data_bunch_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_data_bunch_stream_reader> ptr;

		structured_data_bunch_stream_reader(
			const std::map<std::string, structured_data_reader::ptr>& data_reader_map,
			unsigned int multiple_epoch_count = 1);

		virtual ~structured_data_bunch_stream_reader();

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const;

		// The method returns false in case the entry cannot be read
		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map);

		virtual int get_entry_count() const;

		virtual void next_epoch();

	protected:
		std::map<std::string, structured_data_reader::ptr> data_reader_map;
		int total_entry_count;
		std::vector<int> entry_count_list;
		std::vector<int> base_entry_count_list;
		unsigned int current_epoch;
	};
}
