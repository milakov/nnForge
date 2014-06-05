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

#include "network_data_peeker_load_resume.h"

#include "nn_types.h"

#include <algorithm>

#include <boost/format.hpp>
#include <boost/filesystem/fstream.hpp>

namespace nnforge
{
	const char * network_data_peeker_load_resume::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)\\.data$";
	const char * network_data_peeker_load_resume::resume_ann_index_extractor_pattern = "^ann_trained_(\\d+)_epoch_(\\d+)\\.data$";

	network_data_peeker_load_resume::network_data_peeker_load_resume(
		const boost::filesystem::path& trained_ann_folder_path,
		const boost::filesystem::path& resume_ann_folder_path)
	{
		std::set<unsigned int> trained_ann_list = get_trained_ann_list(trained_ann_folder_path);

		std::map<unsigned int, unsigned int> resume_ann_list = get_resume_ann_list(resume_ann_folder_path, trained_ann_list);

		for(std::map<unsigned int, unsigned int>::const_iterator it = resume_ann_list.begin(); it != resume_ann_list.end(); ++it)
		{
			network_data_peek_entry new_item;
			new_item.index = it->first;
			new_item.start_epoch = it->second;
			std::string filename = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|.data") % new_item.index % new_item.start_epoch).str();
			boost::filesystem::path filepath = resume_ann_folder_path / filename;
			new_item.data = network_data_smart_ptr(new network_data());
			{
				boost::filesystem::ifstream in(filepath, std::ios_base::in | std::ios_base::binary);
				new_item.data->read(in);
			}

			entry_list.push_back(new_item);
		}

		std::sort(entry_list.begin(), entry_list.end(), compare_entry);
	}

	network_data_peeker_load_resume::~network_data_peeker_load_resume()
	{
	}

	bool network_data_peeker_load_resume::compare_entry(network_data_peek_entry i, network_data_peek_entry j)
	{
		return (i.index > j.index);
	}

	std::set<unsigned int> network_data_peeker_load_resume::get_trained_ann_list(const boost::filesystem::path& trained_ann_folder_path) const
	{
		std::set<unsigned int> res;
		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(trained_ann_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
					res.insert(index);
				}
			}
		}

		return res;
	}

	std::map<unsigned int, unsigned int> network_data_peeker_load_resume::get_resume_ann_list(
			const boost::filesystem::path& resume_ann_folder_path,
			const std::set<unsigned int>& exclusion_ann_list) const
	{
		std::map<unsigned int, unsigned int> res;
		nnforge_regex expression(resume_ann_index_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(resume_ann_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
					if (exclusion_ann_list.find(index) == exclusion_ann_list.end())
					{
						unsigned int epoch = static_cast<unsigned int>(atol(std::string(what[2].first, what[2].second).c_str()));
						std::map<unsigned int, unsigned int>::iterator it2 = res.find(index);
						if (it2 == res.end())
							res.insert(std::make_pair(index, epoch));
						else
							it2->second = std::max(it2->second, epoch);
					}
				}
			}
		}

		return res;
	}

	network_data_peek_entry network_data_peeker_load_resume::peek(network_schema_smart_ptr schema)
	{
		network_data_peek_entry res;

		if (entry_list.empty())
			return res;

		res = entry_list.back();

		entry_list.pop_back();

		return res;
	}
}
