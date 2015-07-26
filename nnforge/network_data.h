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

#include "layer_data.h"
#include "layer_data_custom.h"
#include "layer.h"
#include "nn_types.h"
#include "rnd.h"
#include "layer_data_list.h"
#include "layer_data_custom_list.h"

#include <vector>
#include <ostream>
#include <istream>
#include <string>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class network_data
	{
	public:
		typedef nnforge_shared_ptr<network_data> ptr;
		typedef nnforge_shared_ptr<const network_data> const_ptr;

		network_data();

		network_data(
			const std::vector<layer::const_ptr>& layer_list,
			float val = 0.0F);

		// Narrow other network_data to the layers from layer_list
		network_data(
			const std::vector<layer::const_ptr>& layer_list,
			const network_data& other);

		const boost::uuids::uuid& get_uuid() const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		// The method throws exception in case the data is not suitable for the layers
		void check_network_data_consistency(const std::vector<layer::const_ptr>& layer_list) const;

		void randomize(
			const std::vector<layer::const_ptr>& layer_list,
			random_generator& gen);

	private:
		void read_legacy(
			std::istream& binary_stream_to_read_from,
			bool read_data_custom);

	public:
		layer_data_list data_list;
		layer_data_custom_list data_custom_list;

	private:
		static const boost::uuids::uuid data_guid;
		static const boost::uuids::uuid data_guid_v2;
		static const boost::uuids::uuid data_guid_v1;
	};
}
