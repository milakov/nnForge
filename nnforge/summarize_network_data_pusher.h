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

#include "network_data_pusher.h"

#include <boost/filesystem.hpp>

namespace nnforge
{
	class summarize_network_data_pusher : public network_data_pusher
	{
	public:
		summarize_network_data_pusher(const boost::filesystem::path& folder_path);

		virtual ~summarize_network_data_pusher() = default;

		virtual void push(
			const training_task_state& task_state,
			const network_schema& schema);

	private:
		boost::filesystem::path folder_path;
	};
}
