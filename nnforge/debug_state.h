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

#include <boost/filesystem.hpp>
#include <mutex>
#include <memory>

namespace nnforge
{
	class debug_state
	{
	public:
		typedef std::shared_ptr<debug_state> ptr;

		debug_state(
			bool debug_mode,
			const boost::filesystem::path& debug_folder);

		~debug_state() = default;

		bool is_debug() const;

		boost::filesystem::path get_path_to_unique_file(
			const char * file_prefix,
			const char * file_extension);

		void output_message(const char * msg);

	protected:
		bool debug_mode;
		boost::filesystem::path debug_folder;

	private:
		std::mutex index_mutex;
		unsigned int index;

	private:
		debug_state() = delete;
		debug_state(const debug_state&) = delete;
		debug_state& operator =(const debug_state&) = delete;
	};
}
