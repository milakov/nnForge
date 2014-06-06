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

#include "stream_duplicator.h"

#include <iostream>

namespace nnforge
{
	stream_duplicator::stream_duplicator(const boost::filesystem::path& logfile_path)
		: logfile_stream(logfile_path, std::ios_base::out | std::ios_base::app)
		, cout_stream(std::cout.rdbuf())
		, td(cout_stream, logfile_stream)
		, ts(td)
	{
		std::cout.rdbuf(ts.rdbuf());
		std::cout << "########################################" << std::endl;
	}

	stream_duplicator::~stream_duplicator()
	{
		std::cout << "########################################" << std::endl;
		ts.flush();
		std::cout.rdbuf(cout_stream.rdbuf());
		ts.close();
		td.close();
		logfile_stream.close();
	}
}
