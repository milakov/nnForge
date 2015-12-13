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

#include "save_snapshot_network_data_pusher.h"

#include "neural_network_exception.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	save_snapshot_network_data_pusher::save_snapshot_network_data_pusher(const boost::filesystem::path& folder_path)
		: folder_path(folder_path)
	{
	}

	save_snapshot_network_data_pusher::~save_snapshot_network_data_pusher()
	{
	}

	void save_snapshot_network_data_pusher::push(
		const training_task_state& task_state,
		const network_schema& schema)
	{
		unsigned int index = task_state.index_peeked;

		std::string data_folder_name = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|") % index % task_state.get_current_epoch()).str();
		save_data_to_file(task_state.data, data_folder_name);

		{
			std::string momentum_data_folder_name = (boost::format("momentum_%|1$03d|") % index).str();
			if (task_state.momentum_data)
				save_data_to_file(task_state.momentum_data, momentum_data_folder_name);
			else
				boost::filesystem::remove_all(folder_path / momentum_data_folder_name);
		}

		{
			std::string momentum2_data_folder_name = (boost::format("momentum2_%|1$03d|") % index).str();
			if (task_state.momentum_data2)
				save_data_to_file(task_state.momentum_data2, momentum2_data_folder_name);
			else
				boost::filesystem::remove_all(folder_path / momentum2_data_folder_name);
		}
	}

	void save_snapshot_network_data_pusher::save_data_to_file(
		network_data::const_ptr data,
		std::string folder_name) const
	{
		std::string temp_folder_name = folder_name + ".temp";

		boost::filesystem::path temp_full_folder_path = folder_path / temp_folder_name;
		data->write(temp_full_folder_path);

		boost::filesystem::path full_folder_path = folder_path / folder_name;
		if (boost::filesystem::exists(full_folder_path))
			boost::filesystem::remove_all(full_folder_path);

		boost::filesystem::rename(temp_full_folder_path, full_folder_path);
	}
}
