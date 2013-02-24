/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include <vector>
#include <ostream>
#include <istream>
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

		// The result includes input configuration
		layer_configuration_specific_list get_layer_configuration_specific_list(const layer_configuration_specific& input_layer_configuration_specific) const;

		operator const const_layer_list&() const;

	private:
		void clear();

		const_layer_list layers;
		layer_configuration output_config;

		static const boost::uuids::uuid schema_guid;
	};

	typedef std::tr1::shared_ptr<network_schema> network_schema_smart_ptr;
}
