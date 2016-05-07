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
#include "neuron_value_set.h"

#include <map>

namespace nnforge
{
	class neuron_value_set_data_bunch_writer : public structured_data_bunch_writer
	{
	public:
		typedef nnforge_shared_ptr<neuron_value_set_data_bunch_writer> ptr;

		neuron_value_set_data_bunch_writer();

		~neuron_value_set_data_bunch_writer();

		virtual void set_config_map(const std::map<std::string, layer_configuration_specific> config_map);

		virtual void write(
			unsigned int entry_id,
			const std::map<std::string, const float *>& data_map);

	public:
		std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> > layer_name_to_config_and_value_set_map;
	};
}
