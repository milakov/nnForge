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

#include "structured_data_bunch_reader.h"

namespace nnforge
{
	structured_data_bunch_reader::structured_data_bunch_reader()
	{
	}

	structured_data_bunch_reader::~structured_data_bunch_reader()
	{
	}

	int structured_data_bunch_reader::get_entry_count() const
	{
		return -1;
	}

	structured_data_bunch_reader::ptr structured_data_bunch_reader::get_narrow_reader(const std::set<std::string>& layer_names) const
	{
		return structured_data_bunch_reader::ptr();
	}
}
