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

#include "layer_configuration_specific.h"
#include "nn_types.h"

#include <map>

namespace nnforge
{
	class structured_data_bunch_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_data_bunch_reader> ptr;

		virtual ~structured_data_bunch_reader();

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const = 0;

		// The method returns false in case the entry cannot be read
		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map) = 0;

		virtual void next_epoch() const = 0;

		// Return -1 in case there is no info on entry count
		virtual int get_entry_count() const;

	protected:
		structured_data_bunch_reader();

	private:
		structured_data_bunch_reader(const structured_data_bunch_reader&);
		structured_data_bunch_reader& operator =(const structured_data_bunch_reader&);
	};
}
