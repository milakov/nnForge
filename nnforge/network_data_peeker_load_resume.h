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

#pragma once

#include "network_data_peeker.h"

#include <boost/filesystem.hpp>
#include <set>
#include <map>

namespace nnforge
{
	class network_data_peeker_load_resume : public network_data_peeker
	{
	public:
		network_data_peeker_load_resume(
			const boost::filesystem::path& trained_ann_folder_path,
			const boost::filesystem::path& resume_ann_folder_path);

		virtual ~network_data_peeker_load_resume();

		// The method should return empty data smart pointer in case no more layer data are available
		// The caller is free to modify the data returned
		virtual network_data_peek_entry peek(network_schema_smart_ptr schema);

	protected:
		std::set<unsigned int> get_trained_ann_list(const boost::filesystem::path& trained_ann_folder_path) const;

		std::map<unsigned int, unsigned int> get_resume_ann_list(
			 const boost::filesystem::path& resume_ann_folder_path,
			 const std::set<unsigned int>& exclusion_ann_list) const;

		std::vector<network_data_peek_entry> entry_list;

	private:
		static const char * trained_ann_index_extractor_pattern;
		static const char * resume_ann_index_extractor_pattern;

		static bool compare_entry(network_data_peek_entry i, network_data_peek_entry j);
	};
}
