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
#include "feature_map_data_stat.h"
#include "nn_types.h"
#include "raw_data_reader.h"

namespace nnforge
{
	class structured_data_reader : public raw_data_reader
	{
	public:
		typedef nnforge_shared_ptr<structured_data_reader> ptr;

		virtual ~structured_data_reader();

		virtual bool read(
			unsigned int entry_id,
			float * data) = 0;

		virtual bool raw_read(
			unsigned int entry_id,
			std::vector<unsigned char>& all_elems);

		virtual layer_configuration_specific get_configuration() const = 0;

		std::vector<feature_map_data_stat> get_feature_map_data_stat_list();

	protected:
		structured_data_reader();

	private:
		structured_data_reader(const structured_data_reader&);
		structured_data_reader& operator =(const structured_data_reader&);
	};
}
