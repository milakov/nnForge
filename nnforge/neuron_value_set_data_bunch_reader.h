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

#include "layer_configuration_specific.h"
#include "nn_types.h"
#include "structured_data_bunch_reader.h"
#include "neuron_value_set.h"

#include <map>

namespace nnforge
{
	class neuron_value_set_data_bunch_reader : public structured_data_bunch_reader
	{
	public:
		typedef nnforge_shared_ptr<neuron_value_set_data_bunch_reader> ptr;

		neuron_value_set_data_bunch_reader(
			const std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >& layer_name_to_config_and_value_set_map);

		virtual ~neuron_value_set_data_bunch_reader();

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const;

		// The method returns false in case the entry cannot be read
		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map);

		virtual void set_epoch(unsigned int epoch_id);

		// Return -1 in case there is no info on entry count
		virtual int get_entry_count() const;

		// Empty return value (default) indicates original reader should be used
		virtual structured_data_bunch_reader::ptr get_narrow_reader(const std::set<std::string>& layer_names) const;

	public:
		std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> > layer_name_to_config_and_value_set_map;
	};
}
