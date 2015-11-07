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

#include "stream_redirector.h"

#include <iostream>

namespace nnforge
{
	stream_redirector::stream_redirector(const boost::filesystem::path& logfile_path)
		: logfile_stream(logfile_path, std::ios_base::out | std::ios_base::app)
		, backup(std::cout.rdbuf())
	{
		std::cout.rdbuf(logfile_stream.rdbuf());
		std::cout << "########################################" << std::endl;
	}

	stream_redirector::~stream_redirector()
	{
		std::cout << "########################################" << std::endl;
		std::cout.rdbuf(backup);
		logfile_stream.close();
	}
}
