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

#include "layer.h"
#include "layer_configuration.h"
#include "layer_configuration_specific.h"
#include "layer_data_configuration.h"
#include "nn_types.h"

#include <vector>
#include <ostream>
#include <istream>
#include <string>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class network_schema
	{
	public:
		network_schema();

		const boost::uuids::uuid& get_uuid() const;

		const const_layer_list& get_layers() const;

		void add_layer(const_layer_smart_ptr new_layer);

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		void write_proto(std::ostream& stream_to_write_to) const;

		void read_proto(std::istream& stream_to_read_from);

		// The result includes input configuration
		layer_configuration_specific_list get_layer_configuration_specific_list(const layer_configuration_specific& input_layer_configuration_specific) const;

		// The result includes output configuration
		layer_configuration_specific_list get_layer_configuration_specific_list_reverse(const layer_configuration_specific& output_layer_configuration_specific) const;

		// Returns minimal input rectangle which is quasi-transformed into output one covering the rectangle supplied
		std::vector<std::pair<unsigned int, unsigned int> > get_input_rectangle_borders(
			const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders,
			unsigned int output_layer_id) const;

		std::vector<layer_data_configuration_list> get_layer_data_configuration_list_list() const;

		operator const const_layer_list&() const;

	public:
		std::string name;

	private:
		void clear();

		const_layer_list layers;
		layer_configuration output_config;

		static const boost::uuids::uuid schema_guid;
	};

	typedef nnforge_shared_ptr<network_schema> network_schema_smart_ptr;
	typedef nnforge_shared_ptr<const network_schema> const_network_schema_smart_ptr;
}
