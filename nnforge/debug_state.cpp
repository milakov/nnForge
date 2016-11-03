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

#include "debug_state.h"

#include <boost/format.hpp>
#include <iostream>

namespace nnforge
{
	debug_state::debug_state(
		bool debug_mode,
		const boost::filesystem::path& debug_folder)
		: debug_mode(debug_mode)
		, index(0)
	{
		if (debug_mode)
		{
			{
				time_t rawtime;
				struct tm * timeinfo;
				char buffer[80];
				time(&rawtime);
				timeinfo = localtime(&rawtime);
				strftime(buffer, 80, "%Y-%m-%d_%H%M%S", timeinfo);
				this->debug_folder = debug_folder / buffer;
			}

			boost::filesystem::create_directories(this->debug_folder);

			output_message((boost::format("Debug files will be saved into %1%") % this->debug_folder.string()).str().c_str());
		}
	}

	bool debug_state::is_debug() const
	{
		return debug_mode;
	}

	boost::filesystem::path debug_state::get_path_to_unique_file(
		const char * file_prefix,
		const char * file_extension)
	{
		unsigned int old_val;
		{
			std::lock_guard<std::mutex> lock(index_mutex);
			old_val = index++;
		}
		boost::filesystem::path file_path = debug_folder / (boost::format("%|1$05d|_%2%.%3%") % old_val % file_prefix % file_extension).str();
		return file_path;
	}

	void debug_state::output_message(const char * msg)
	{
		if (debug_mode)
			std::cout << "DEBUG: " << msg << std::endl;
	}
}
