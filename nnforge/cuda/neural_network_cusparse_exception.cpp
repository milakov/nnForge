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

#include "neural_network_cusparse_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		neural_network_cusparse_exception::neural_network_cusparse_exception(
			cusparseStatus_t error_code,
			const char * filename,
			int line_number)
			: neural_network_exception((boost::format("cuSPARSE error: %1%") % error_code).str(), filename, line_number)
		{
		}
	}
}
