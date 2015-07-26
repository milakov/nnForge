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
#include "rnd.h"

#include <map>
#include <string>
#include <ostream>
#include <istream>

namespace nnforge
{
	class layer_data_list
	{
	public:
		typedef nnforge_shared_ptr<layer_data_list> ptr;
		typedef nnforge_shared_ptr<const layer_data_list> const_ptr;

		layer_data_list();

		layer_data_list(
			const std::vector<layer::const_ptr>& layer_list,
			float val = 0.0F);

		// Narrow other data to the layers from layer_list
		layer_data_list(
			const std::vector<layer::const_ptr>& layer_list,
			const layer_data_list& other);

		void check_consistency(const std::vector<layer::const_ptr>& layer_list) const;

		std::string get_stat() const;

		void fill(float val);

		void random_fill(
			float min,
			float max,
			random_generator& gen);

		// Returns empty smart pointer in case no data is associated with instance_name
		layer_data::ptr find(const std::string& instance_name) const;

		layer_data::ptr get(const std::string& instance_name) const;

		void add(
			const std::string& instance_name,
			layer_data::ptr data);

		void write(std::ostream& binary_stream_to_write_to) const;

		void read(std::istream& binary_stream_to_read_from);

	private:
		std::map<std::string, layer_data::ptr> instance_name_to_data_map;
	};
}
