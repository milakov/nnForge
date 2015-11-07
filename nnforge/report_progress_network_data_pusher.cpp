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

#include "report_progress_network_data_pusher.h"

#include <stdio.h>
#include <boost/format.hpp>

namespace nnforge
{
	report_progress_network_data_pusher::report_progress_network_data_pusher()
	{
	}

	report_progress_network_data_pusher::~report_progress_network_data_pusher()
	{
	}

	void report_progress_network_data_pusher::push(
		const training_task_state& task_state,
		const network_schema& schema)
	{
		unsigned int last_index = static_cast<unsigned int>(task_state.history.size()) - 1;

		std::cout << "----- Training -----" << std::endl;
		std::cout << task_state.history[last_index].first << std::endl;
		if (!task_state.comments[last_index].empty())
			std::cout << task_state.comments[last_index] << std::endl;

		std::cout << "Avg [rate weights updates]";
		std::vector<std::string> data_name_list = task_state.data->data_list.get_data_layer_name_list();
		for(std::vector<std::string>::const_iterator it = data_name_list.begin(); it != data_name_list.end(); ++it)
		{
			layer_data::ptr layer_data = task_state.data->data_list.get(*it);
			if (!layer_data->empty())
			{
				std::cout << ", " << *it;
				const std::vector<float>& absolute_updates = task_state.history[last_index].first.average_absolute_updates.find(*it)->second;
				for(int part_id = 0; part_id < layer_data->size(); ++part_id)
				{
					const std::vector<float>& weights = layer_data->at(part_id);
					double sum = 0.0;
					for(std::vector<float>::const_iterator it = weights.begin(); it != weights.end(); ++it)
						sum += static_cast<double>(fabsf(*it));
					float avg_weight = static_cast<float>(sum) / static_cast<float>(weights.size());

					std::cout << (boost::format(" [%|1$.2e| %|2$.2e| %|3$.2e|]") % (absolute_updates[part_id] / avg_weight) % avg_weight % absolute_updates[part_id]); 
				}
			}
		}
		std::cout << std::endl;

		for(std::map<std::string, std::pair<layer_configuration_specific, nnforge_shared_ptr<std::vector<float> > > >::const_iterator it = task_state.history[last_index].second.begin(); it != task_state.history[last_index].second.end(); ++it)
			std::cout << schema.get_layer(it->first)->get_string_for_average_data(it->second.first, *it->second.second) << std::endl;
	}
}
