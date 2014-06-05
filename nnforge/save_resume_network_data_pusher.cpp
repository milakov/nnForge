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

#include "save_resume_network_data_pusher.h"

#include "neural_network_exception.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	save_resume_network_data_pusher::save_resume_network_data_pusher(const boost::filesystem::path& folder_path)
		: folder_path(folder_path)
	{
	}

	save_resume_network_data_pusher::~save_resume_network_data_pusher()
	{
	}

	void save_resume_network_data_pusher::push(const training_task_state& task_state)
	{
		unsigned int index = task_state.index_peeked;
		network_data_smart_ptr data = task_state.data;

		std::string filename = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|.data") % index % task_state.get_current_epoch()).str();
		std::string temp_filename = filename + ".temp";

		boost::filesystem::path filepath = folder_path / filename;
		boost::filesystem::path temp_filepath = folder_path / temp_filename;

		{
			boost::filesystem::ofstream file_with_data(temp_filepath, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			data->write(file_with_data);
		}

		boost::filesystem::rename(temp_filepath, filepath);
	}
}
