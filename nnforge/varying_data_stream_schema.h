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

#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class varying_data_stream_schema
	{
	public:
		static const boost::uuids::uuid varying_data_stream_guid;

	private:
		varying_data_stream_schema() = delete;
		varying_data_stream_schema(const varying_data_stream_schema&) = delete;
		varying_data_stream_schema& operator =(const varying_data_stream_schema&) = delete;
	};
}
