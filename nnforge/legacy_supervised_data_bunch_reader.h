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
#include "neuron_data_type.h"

#include <string>
#include <boost/thread/thread.hpp>

namespace nnforge
{
	class legacy_supervised_data_bunch_reader : public structured_data_bunch_reader
	{
	public:
		typedef nnforge_shared_ptr<legacy_supervised_data_bunch_reader> ptr;

		legacy_supervised_data_bunch_reader(
			nnforge_shared_ptr<std::istream> input_stream,
			const char * input_data_layer_name,
			const char * output_data_layer_name);

		virtual ~legacy_supervised_data_bunch_reader();

		virtual std::map<std::string, layer_configuration_specific> get_config_map() const;

		// The method returns false in case the entry cannot be read
		virtual bool read(
			unsigned int entry_id,
			const std::map<std::string, float *>& data_map);

		virtual void next_epoch() const;

		virtual int get_approximate_entry_count() const;

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		std::string input_data_layer_name;
		std::string output_data_layer_name;

		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		neuron_data_type::input_type type_code;
		unsigned int entry_count;

		std::istream::pos_type reset_pos;

		boost::mutex read_data_mutex;
	};
}
