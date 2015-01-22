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

#pragma once

#include "network_data_pusher.h"

#include "network_tester.h"
#include "supervised_data_reader.h"
#include "output_neuron_value_set.h"
#include "testing_complete_result_set_visualizer.h"
#include "error_function.h"

namespace nnforge
{
	class validate_progress_network_data_pusher : public network_data_pusher
	{
	public:
		validate_progress_network_data_pusher(
			network_tester_smart_ptr tester,
			supervised_data_reader_smart_ptr reader,
			testing_complete_result_set_visualizer_smart_ptr visualizer,
			const_error_function_smart_ptr ef,
			unsigned int sample_count,
			unsigned int report_frequency = 1);

		virtual ~validate_progress_network_data_pusher();

		virtual void push(const training_task_state& task_state);

	protected:
		network_tester_smart_ptr tester;
		supervised_data_reader_smart_ptr reader;
		output_neuron_value_set_smart_ptr actual_output_neuron_value_set;
		testing_complete_result_set_visualizer_smart_ptr visualizer;
		const_error_function_smart_ptr ef;
		unsigned int sample_count;
		unsigned int report_frequency;
	};
}
