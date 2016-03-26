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
#include <boost/thread/thread.hpp>

#include "nn_types.h"

namespace nnforge
{
	class profile_state
	{
	public:
		typedef nnforge_shared_ptr<profile_state> ptr;

		profile_state(
			bool profile_mode,
			const boost::filesystem::path& profile_folder);

		~profile_state();

		bool is_profile() const;

		boost::filesystem::path get_path_to_unique_file(
			const char * file_prefix,
			const char * file_extension);

		void output_message(const char * msg);

	protected:
		bool profile_mode;
		boost::filesystem::path profile_folder;

	private:
		boost::mutex index_mutex;
		unsigned int index;

	private:
		profile_state();
		profile_state(const profile_state&);
		profile_state& operator =(const profile_state&);
	};
}
