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

#pragma once

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <ostream>

namespace nnforge
{
	class stream_redirector
	{
	public:
		stream_redirector(const boost::filesystem::path& logfile_path);

		~stream_redirector();

	private:
		boost::filesystem::ofstream logfile_stream;
		std::streambuf * backup;
	};
}
