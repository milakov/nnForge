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

#include "layer_data.h"
#include "layer_data_custom.h"
#include "layer.h"
#include "rnd.h"
#include "layer_data_list.h"
#include "layer_data_custom_list.h"

#include <vector>
#include <string>
#include <memory>
#include <boost/uuid/uuid.hpp>
#include <boost/filesystem.hpp>

namespace nnforge
{
	class network_data
	{
	public:
		typedef std::shared_ptr<network_data> ptr;
		typedef std::shared_ptr<const network_data> const_ptr;

		network_data() = default;

		network_data(
			const std::vector<layer::const_ptr>& layer_list,
			float val = 0.0F);

		// Narrow other network_data to the layers from layer_list
		network_data(
			const std::vector<layer::const_ptr>& layer_list,
			const network_data& other);

		const boost::uuids::uuid& get_uuid() const;

		void write(const boost::filesystem::path& folder_path) const;

		void read(const boost::filesystem::path& folder_path);

		// The method throws exception in case the data is not suitable for the layers
		void check_network_data_consistency(const std::vector<layer::const_ptr>& layer_list) const;

		void randomize(
			const std::vector<layer::const_ptr>& layer_list,
			random_generator& gen);

	public:
		layer_data_list data_list;
		layer_data_custom_list data_custom_list;

	private:
		static const boost::uuids::uuid data_guid;
	};
}
