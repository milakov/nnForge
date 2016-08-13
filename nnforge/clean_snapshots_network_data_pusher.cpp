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

#include "clean_snapshots_network_data_pusher.h"

#include "neural_network_exception.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	const char * clean_snapshots_network_data_pusher::snapshot_ann_index_extractor_pattern = "^ann_trained_(\\d+)_epoch_(\\d+)$";
	
	clean_snapshots_network_data_pusher::clean_snapshots_network_data_pusher(
		const boost::filesystem::path& folder_path,
		unsigned int keep_frequency)
		: folder_path(folder_path)
		, keep_frequency(keep_frequency)
	{
	}

	clean_snapshots_network_data_pusher::~clean_snapshots_network_data_pusher()
	{
	}

	void clean_snapshots_network_data_pusher::push(
		const training_task_state& task_state,
		const network_schema& schema)
	{
		unsigned int previous_epoch = task_state.get_current_epoch() - 1;

		if (previous_epoch % keep_frequency != 0)
		{
			unsigned int current_index = task_state.index_peeked;
			std::string snapshot_folder_name = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|") % current_index % previous_epoch).str();
			boost::filesystem::path folder_path_to_clean = folder_path / snapshot_folder_name;
			if (boost::filesystem::exists(folder_path_to_clean))
				boost::filesystem::remove_all(folder_path_to_clean);
		}
	}
}
