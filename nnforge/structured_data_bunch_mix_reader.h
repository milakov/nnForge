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

namespace nnforge
{
	class structured_data_bunch_mix_reader : public structured_data_bunch_reader
	{
	public:
		typedef std::shared_ptr<structured_data_bunch_mix_reader> ptr;

		structured_data_bunch_mix_reader(
			structured_data_bunch_reader::ptr main_reader,
			structured_data_bunch_reader::ptr auxiliary_reader,
			float auxiliary_reader_part);

		~structured_data_bunch_mix_reader() = default;

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const;

		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map);

		virtual int get_entry_count() const;

		virtual structured_data_bunch_reader::ptr get_narrow_reader(const std::set<std::string>& layer_names) const;

		virtual void set_epoch(unsigned int epoch_id);

	protected:
		void update_redirect_entry_list();

	protected:
		structured_data_bunch_reader::ptr main_reader;
		structured_data_bunch_reader::ptr auxiliary_reader;
		float auxiliary_reader_part;

		std::vector<int> redirect_entry_list;
	};
}
