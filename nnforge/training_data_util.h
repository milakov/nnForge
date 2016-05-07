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

#include "structured_data_bunch_writer.h"
#include "structured_data_bunch_reader.h"

#include <set>
#include <string>

namespace nnforge
{
	class training_data_util
	{
	public:
		static void copy(
			const std::set<std::string>& layers_to_copy,
			structured_data_bunch_writer& writer,
			structured_data_bunch_reader& reader,
			int max_copy_elem_count = -1);

	private:
		training_data_util();
		~training_data_util();
	};
}
