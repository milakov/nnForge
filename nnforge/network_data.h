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

#include "layer_data.h"
#include "layer.h"
#include "dropout_layer_config.h"
#include "nn_types.h"
#include "rnd.h"

#include <vector>
#include <ostream>
#include <istream>
#include <string>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class network_data : public layer_data_list
	{
	public:
		network_data();

		const boost::uuids::uuid& get_uuid() const;

		// All data values are initialized to 0.0F
		network_data(const const_layer_list& layer_list, float val = 0.0F);

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		// The method throws exception in case the data is not suitable for the layers
		void check_network_data_consistency(const const_layer_list& layer_list) const;

		void randomize(
			const const_layer_list& layer_list,
			random_generator& gen);

		void fill(float val);

		void random_fill(
			float min,
			float max,
			random_generator& gen);

		void apply_dropout_layer_config(
			const std::map<unsigned int, dropout_layer_config>& layer_id_to_dropout_config_map,
			bool is_direct);

		std::string get_stat() const;

	private:
		static const boost::uuids::uuid data_guid;
	};

	typedef nnforge_shared_ptr<network_data> network_data_smart_ptr;
	typedef nnforge_shared_ptr<const network_data> network_data_const_smart_ptr;
}
