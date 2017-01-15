/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <cassert>
#include <iostream>

namespace nnforge
{
	neural_network_exception::neural_network_exception(const char * message)
		: std::runtime_error(message)
	{
#ifndef NDEBUG
		std::cerr << std::runtime_error::what() << std::endl;
#endif
		assert(0);
	}

	neural_network_exception::neural_network_exception(const std::string& message)
		: std::runtime_error(message)
	{
#ifndef NDEBUG
		std::cerr << std::runtime_error::what() << std::endl;
#endif
		assert(0);
	}

	neural_network_exception::neural_network_exception(
		const std::string& message,
		const char * filename,
		int line_number)
		: std::runtime_error((boost::format("%1% in %2%:%3%") % message % filename % line_number).str())
	{
#ifndef NDEBUG
		std::cerr << std::runtime_error::what() << std::endl;
#endif
		assert(0);
	}
}
