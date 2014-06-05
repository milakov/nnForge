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

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <ostream>

namespace nnforge
{
	class stream_duplicator
	{
	public:
		stream_duplicator(const boost::filesystem::path& logfile_path);

		~stream_duplicator();

	private:
		boost::filesystem::ofstream logfile_stream;
		std::ostream cout_stream;
		boost::iostreams::tee_device<std::ostream, boost::filesystem::ofstream> td;
		boost::iostreams::stream<boost::iostreams::tee_device<std::ostream, boost::filesystem::ofstream> > ts;
	};
}
