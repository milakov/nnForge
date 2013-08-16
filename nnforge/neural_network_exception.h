/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include <stdexcept>

namespace nnforge
{
	class neural_network_exception : public std::runtime_error
	{
	public:
		neural_network_exception(const char * message);

		neural_network_exception(const std::string& message);

		neural_network_exception(
			const std::string& message,
			const char * filename,
			int line_number);
	};
}
