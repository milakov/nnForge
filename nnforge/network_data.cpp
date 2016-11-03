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

#include "network_data.h"

#include "neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>
#include <numeric> 

namespace nnforge
{
	// {4FF032B3-EA2B-481D-A278-FDA9AFEBE76D}
	const boost::uuids::uuid network_data::data_guid =
		{ 0x4f, 0xf0, 0x32, 0xb3
		, 0xea, 0x2b
		, 0x48, 0x1d
		, 0xa2, 0x78
		, 0xfd, 0xa9, 0xaf, 0xeb, 0xe7, 0x6d };

	network_data::network_data(
		const std::vector<layer::const_ptr>& layer_list,
		float val)
		: data_list(layer_list, val)
		, data_custom_list(layer_list)
	{
	}

	network_data::network_data(
		const std::vector<layer::const_ptr>& layer_list,
		const network_data& other)
		: data_list(layer_list, other.data_list)
		, data_custom_list(layer_list, other.data_custom_list)
	{
	}

	void network_data::check_network_data_consistency(const std::vector<layer::const_ptr>& layer_list) const
	{
		data_list.check_consistency(layer_list);
		data_custom_list.check_consistency(layer_list);
	}

	const boost::uuids::uuid& network_data::get_uuid() const
	{
		return data_guid;
	}

	void network_data::write(const boost::filesystem::path& folder_path) const
	{
		data_list.write(folder_path);
		data_custom_list.write(folder_path);
	}

	void network_data::read(const boost::filesystem::path& folder_path)
	{
		data_list.read(folder_path);
		data_custom_list.read(folder_path);
	}

	void network_data::randomize(
		const std::vector<layer::const_ptr>& layer_list,
		random_generator& gen)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer_data::ptr data = data_list.find((*it)->instance_name);
			layer_data_custom::ptr data_custom = data_custom_list.find((*it)->instance_name);
			(*it)->randomize_data(
				data,
				data_custom,
				gen);
		}
	}
}
