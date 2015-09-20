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

#include "layer_data_custom_list.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace nnforge
{
	// {049EB47D-AF04-4957-8E6A-7CBC4C4D1754}
	const boost::uuids::uuid layer_data_custom_list::data_custom_guid =
		{ 0x04, 0x9e, 0xb4, 0x7d
		, 0xaf, 0x04
		, 0x49, 0x57
		, 0x8e, 0x6a
		, 0x7c, 0xbc, 0x4c, 0x4d, 0x17, 0x54 };

	const char * layer_data_custom_list::data_custom_extractor_pattern = "^(.+)\\.datac$";

	layer_data_custom_list::layer_data_custom_list()
	{
	}

	layer_data_custom_list::layer_data_custom_list(const std::vector<layer::const_ptr>& layer_list)
	{
		instance_name_to_data_custom_map.clear();
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			layer_data_custom::ptr data_custom = l->create_layer_data_custom();
			if (!data_custom->empty())
				add(l->instance_name, data_custom);
		}
	}

	layer_data_custom_list::layer_data_custom_list(
		const std::vector<layer::const_ptr>& layer_list,
		const layer_data_custom_list& other)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data_custom())
			{
				layer_data_custom::ptr data_custom = other.find(l->instance_name);
				if (data_custom)
					add(l->instance_name, data_custom);
			}
		}
	}

	void layer_data_custom_list::check_consistency(const std::vector<layer::const_ptr>& layer_list) const
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data_custom())
			{
				layer_data_custom::ptr data_custom = find(l->instance_name);
				if (!data_custom)
					throw neural_network_exception((boost::format("No custom data for layer with name %1%") % l->instance_name).str());
			
				l->check_layer_data_custom_consistency(*data_custom);
			}
		}
	}

	layer_data_custom::ptr layer_data_custom_list::find(const std::string& instance_name) const
	{
		std::map<std::string, layer_data_custom::ptr>::const_iterator it = instance_name_to_data_custom_map.find(instance_name);

		if (it != instance_name_to_data_custom_map.end())
			return it->second;
		else
			return layer_data_custom::ptr();
	}

	layer_data_custom::ptr layer_data_custom_list::get(const std::string& instance_name) const
	{
		layer_data_custom::ptr res = find(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("No custom data found with layer name %1%") % instance_name).str());

		return res;
	}

	void layer_data_custom_list::write(const boost::filesystem::path& folder_path) const
	{
		boost::filesystem::create_directories(folder_path);

		for(std::map<std::string, layer_data_custom::ptr>::const_iterator it = instance_name_to_data_custom_map.begin(); it != instance_name_to_data_custom_map.end(); ++it)
		{
			boost::filesystem::path file_path = folder_path / (it->first + ".datac");
			boost::filesystem::ofstream out(file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			out.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);
			out.write(reinterpret_cast<const char*>(data_custom_guid.data), sizeof(data_custom_guid.data));
			it->second->write(out);
		}
	}

	void layer_data_custom_list::read(const boost::filesystem::path& folder_path)
	{
		instance_name_to_data_custom_map.clear();

		if (!boost::filesystem::exists(folder_path) || !boost::filesystem::is_directory(folder_path))
			throw neural_network_exception((boost::format("Directory %1% doesn't exist") % folder_path).str());

		nnforge_regex expression(data_custom_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					std::string data_name = std::string(what[1].first, what[1].second);
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					in.exceptions(std::istream::eofbit | std::istream::failbit | std::istream::badbit);
					boost::uuids::uuid data_custom_guid_read;
					in.read(reinterpret_cast<char*>(data_custom_guid_read.data), sizeof(data_custom_guid_read.data));
					if (data_custom_guid_read != data_custom_guid)
						throw neural_network_exception((boost::format("Unknown data GUID encountered in input stream: %1%") % data_custom_guid_read).str());
					layer_data_custom::ptr d(new layer_data_custom());
					d->read(in);
					add(data_name, d);
				}
			}
		}
	}

	void layer_data_custom_list::add(
		const std::string& instance_name,
		layer_data_custom::ptr data_custom)
	{
		if (!instance_name_to_data_custom_map.insert(std::make_pair(instance_name, data_custom)).second)
			throw neural_network_exception((boost::format("Custom data with duplicate layer name %1%") % instance_name).str());
	}
}