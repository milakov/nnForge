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

#include "network_trainer_sgd.h"

#include <boost/format.hpp>
#include <numeric>
#include <limits>

#include "neural_network_exception.h"
#include "neuron_value_set_data_bunch_writer.h"

namespace nnforge
{
	network_trainer_sgd::network_trainer_sgd(
		network_schema::ptr schema,
		const std::vector<std::string>& output_layer_names,
		const std::vector<std::string>& error_source_layer_names,
		const std::vector<std::string>& exclude_data_update_layer_names,
		backward_propagation::ptr backprop)
		: network_trainer(schema, output_layer_names, error_source_layer_names, exclude_data_update_layer_names)
		, backprop(backprop)
	{
	}

	void network_trainer_sgd::train_step(
		structured_data_bunch_reader& reader,
		training_task_state& task)
	{
		std::pair<std::map<std::string, std::vector<float> >, std::string> lr_and_comment = prepare_learning_rates(task.get_current_epoch(), task.data);
		task.comments.push_back(lr_and_comment.second);

		neuron_value_set_data_bunch_writer writer;
		backward_propagation::stat training_stat = backprop->run(
			reader,
			writer,
			*task.data,
			task.momentum_data,
			task.momentum_data2,
			lr_and_comment.first,
			batch_size,
			weight_decay,
			momentum,
			task.get_current_epoch());
		std::map<std::string, std::pair<layer_configuration_specific, std::shared_ptr<std::vector<double> > > > output_data_average_results;
		for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = writer.layer_name_to_config_and_value_set_map.begin(); it != writer.layer_name_to_config_and_value_set_map.end(); ++it)
			output_data_average_results.insert(std::make_pair(it->first, std::make_pair(it->second.first, it->second.second->get_average())));

		task.history.push_back(std::make_pair(training_stat, output_data_average_results));
	}

	std::pair<std::map<std::string, std::vector<float> >, std::string> network_trainer_sgd::prepare_learning_rates(
		unsigned int epoch,
		network_data::const_ptr data)
	{
		float learning_rate = get_global_learning_rate(static_cast<unsigned int>(epoch));

		std::map<std::string, std::vector<float> > res;

		std::vector<std::string> layer_names = data->data_list.get_data_layer_name_list();
		for(std::vector<std::string>::const_iterator it = layer_names.begin(); it != layer_names.end(); ++it)
		{
			layer_data::ptr dt = data->data_list.get(*it);

			res.insert(std::make_pair(*it, std::vector<float>(dt->size(), learning_rate)));
		}

		std::string comment = (boost::format("LR %|1$.5e|") % learning_rate).str();

		return std::make_pair(res, comment);
	}

	void network_trainer_sgd::initialize_train(structured_data_bunch_reader& reader)
	{
		backprop->set_input_configuration_specific(reader.get_config_map());
	}
}
