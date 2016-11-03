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

#include <vector>
#include <ostream>
#include <istream>
#include <memory>

namespace nnforge
{
	class layer_data_custom : public std::vector<std::vector<int> >
	{
	public:
		typedef std::shared_ptr<layer_data_custom> ptr;
		typedef std::shared_ptr<const layer_data_custom> const_ptr;

		layer_data_custom() = default;

		// The stream should be created with std::ios_base::binary flag
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		void read(std::istream& binary_stream_to_read_from);
	};
}
