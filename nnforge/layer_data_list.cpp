/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "layer_data_list.h"

#include "neural_network_exception.h"

#include <numeric>
#include <boost/format.hpp>
#include <cmath>
#include <regex>
#include <boost/filesystem/fstream.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace nnforge
{
	// {D11935B5-3059-49E2-9C33-1C4322F8130D}
	const boost::uuids::uuid layer_data_list::data_guid =
		{ 0xd1, 0x19, 0x35, 0xb5
		, 0x30, 0x59
		, 0x49, 0xe2
		, 0x9c, 0x33
		, 0x1c, 0x43, 0x22, 0xf8, 0x13, 0xd };

	const char * layer_data_list::data_extractor_pattern = "^(.+)\\.data$";

	layer_data_list::layer_data_list(
		const std::vector<layer::const_ptr>& layer_list,
		float val)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			layer_data::ptr data = l->create_layer_data();
			if (!data->empty())
			{
				add(l->instance_name, data);
				data->fill(val);
			}
		}
	}

	layer_data_list::layer_data_list(
		const std::vector<layer::const_ptr>& layer_list,
		const layer_data_list& other)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data())
			{
				layer_data::ptr data = other.find(l->instance_name);
				if (data)
					add(l->instance_name, data);
			}
		}
	}

	void layer_data_list::check_consistency(const std::vector<layer::const_ptr>& layer_list) const
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data())
			{
				layer_data::ptr data = find(l->instance_name);
				if (!data)
					throw neural_network_exception((boost::format("No data for layer with name %1%") % l->instance_name).str());
			
				l->check_layer_data_consistency(*data);
			}
		}
	}

	std::string layer_data_list::get_stat() const
	{
		std::string stat = "Avg Abs weights";

		for(std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
		{
			std::string layer_stat;

			for(layer_data::const_iterator it2 = it->second->begin(); it2 != it->second->end(); it2++)
			{
				const std::vector<float>& data = *it2;

				double sum = 0.0;
				for(std::vector<float>::const_iterator it3 = data.begin(); it3 != data.end(); ++it3)
					sum += static_cast<float>(fabsf(*it3));
				float avg = static_cast<float>(sum) / static_cast<float>(data.size());

				if (!layer_stat.empty())
					layer_stat += ", ";
				layer_stat += (boost::format("%|1$.5e|") % avg).str();
			}

			if (!layer_stat.empty())
			{
				stat += (boost::format(" %1% (%2%)") % it->first % layer_stat).str();
			}
		}

		return stat;
	}

	void layer_data_list::fill(float val)
	{
		for(std::map<std::string, layer_data::ptr>::iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
			it->second->fill(val);
	}

	void layer_data_list::random_fill(
		float min,
		float max,
		random_generator& gen)
	{
		for(std::map<std::string, layer_data::ptr>::iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
			it->second->random_fill(min, max, gen);
	}

	layer_data::ptr layer_data_list::find(const std::string& instance_name) const
	{
		std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.find(instance_name);

		if (it != instance_name_to_data_map.end())
			return it->second;
		else
			return layer_data::ptr();
	}

	layer_data::ptr layer_data_list::get(const std::string& instance_name) const
	{
		layer_data::ptr res = find(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("No data found with layer name %1%") % instance_name).str());

		return res;
	}

	void layer_data_list::write(const boost::filesystem::path& folder_path) const
	{
		boost::filesystem::create_directories(folder_path);

		for(std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
		{
			boost::filesystem::path file_path = folder_path / (it->first + ".data");
			boost::filesystem::ofstream out(file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			out.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);
			out.write(reinterpret_cast<const char*>(data_guid.data), sizeof(data_guid.data));
			it->second->write(out);
		}
	}

	void layer_data_list::read(const boost::filesystem::path& folder_path)
	{
		instance_name_to_data_map.clear();

		if (!boost::filesystem::exists(folder_path) || !boost::filesystem::is_directory(folder_path))
			throw neural_network_exception((boost::format("Directory %1% doesn't exist") % folder_path).str());

		std::regex expression(data_extractor_pattern);
		std::cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (std::regex_search(file_name.c_str(), what, expression))
				{
					std::string data_name = std::string(what[1].first, what[1].second);
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					in.exceptions(std::istream::eofbit | std::istream::failbit | std::istream::badbit);
					boost::uuids::uuid data_guid_read;
					in.read(reinterpret_cast<char*>(data_guid_read.data), sizeof(data_guid_read.data));
					if (data_guid_read != data_guid)
						throw neural_network_exception((boost::format("Unknown data GUID encountered in input stream: %1%") % data_guid_read).str());
					layer_data::ptr d(new layer_data());
					d->read(in);
					add(data_name, d);
				}
			}
		}
	}

	void layer_data_list::add(
		const std::string& instance_name,
		layer_data::ptr data)
	{
		if (!instance_name_to_data_map.insert(std::make_pair(instance_name, data)).second)
			throw neural_network_exception((boost::format("Data with duplicate layer name %1%") % instance_name).str());
	}

	std::vector<std::string> layer_data_list::get_data_layer_name_list() const
	{
		std::vector<std::string> res;

		for(std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
			res.push_back(it->first);

		return res;
	}
}
