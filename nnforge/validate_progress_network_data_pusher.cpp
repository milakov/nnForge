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

#include "validate_progress_network_data_pusher.h"

#include <stdio.h>
#include <boost/format.hpp>

namespace nnforge
{
	validate_progress_network_data_pusher::validate_progress_network_data_pusher(
		network_tester_smart_ptr tester,
		supervised_data_reader_smart_ptr reader,
		testing_complete_result_set_visualizer_smart_ptr visualizer,
		const_error_function_smart_ptr ef,
		unsigned int sample_count)
		: tester(tester)
		, reader(reader)
		, visualizer(visualizer)
		, ef(ef)
	{
		actual_output_neuron_value_set = reader->get_output_neuron_value_set(sample_count);
	}

	validate_progress_network_data_pusher::~validate_progress_network_data_pusher()
	{
	}

	void validate_progress_network_data_pusher::push(const training_task_state& task_state)
	{
		tester->set_data(task_state.data);

		testing_complete_result_set testing_res(ef, actual_output_neuron_value_set);
		tester->test(
			*reader,
			testing_res);

		unsigned int last_index = static_cast<unsigned int>(task_state.history.size()) - 1;

		std::cout << "# " << task_state.index_peeked
			<< ", Epoch " << task_state.get_current_epoch()
			<< ", Validating ";
		visualizer->dump(std::cout, testing_res);
		std::cout << std::endl;
	}
}
