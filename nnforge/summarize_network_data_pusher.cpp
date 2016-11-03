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

#include "summarize_network_data_pusher.h"

#include "neural_network_exception.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	summarize_network_data_pusher::summarize_network_data_pusher(const boost::filesystem::path& folder_path)
		: folder_path(folder_path)
	{
	}

	void summarize_network_data_pusher::push(
		const training_task_state& task_state,
		const network_schema& schema)
	{
		unsigned int index = task_state.index_peeked;
		network_data::const_ptr data = task_state.data;

		std::string data_folder_name = (boost::format("ann_trained_%|1$03d|") % index).str();
		data->write(folder_path / data_folder_name);
	}
}
