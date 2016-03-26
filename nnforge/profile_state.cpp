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

#include "profile_state.h"

#include <boost/format.hpp>

namespace nnforge
{
	profile_state::profile_state(
		bool profile_mode,
		const boost::filesystem::path& profile_folder)
		: profile_mode(profile_mode)
		, index(0)
	{
		if (profile_mode)
		{
			{
				time_t rawtime;
				struct tm * timeinfo;
				char buffer[80];
				time(&rawtime);
				timeinfo = localtime(&rawtime);
				strftime(buffer, 80, "%Y-%m-%d_%H%M%S", timeinfo);
				this->profile_folder = profile_folder / buffer;
			}

			boost::filesystem::create_directories(this->profile_folder);

			output_message((boost::format("Profile files will be saved into %1%") % this->profile_folder.string()).str().c_str());
		}
	}

	profile_state::~profile_state()
	{
	}

	bool profile_state::is_profile() const
	{
		return profile_mode;
	}

	boost::filesystem::path profile_state::get_path_to_unique_file(
		const char * file_prefix,
		const char * file_extension)
	{
		unsigned int old_val;
		{
			boost::unique_lock<boost::mutex> lock(index_mutex);
			old_val = index++;
		}
		boost::filesystem::path file_path = profile_folder / (boost::format("%|1$05d|_%2%.%3%") % old_val % file_prefix % file_extension).str();
		return file_path;
	}

	void profile_state::output_message(const char * msg)
	{
		if (profile_mode)
			std::cout << "PROFILE: " << msg << std::endl;
	}
}
