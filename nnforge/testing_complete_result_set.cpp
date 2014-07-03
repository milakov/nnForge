/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "testing_complete_result_set.h"

namespace nnforge
{
	testing_complete_result_set::testing_complete_result_set()
	{
	}

	testing_complete_result_set::testing_complete_result_set(
		const_error_function_smart_ptr ef,
		output_neuron_value_set_smart_ptr actual_output_neuron_value_set)
		: ef(ef)
		, actual_output_neuron_value_set(actual_output_neuron_value_set)
		, predicted_output_neuron_value_set(
			new output_neuron_value_set(
				static_cast<unsigned int>(actual_output_neuron_value_set->neuron_value_list.size()),
				static_cast<unsigned int>(actual_output_neuron_value_set->neuron_value_list.begin()->size())))
	{
	}

	void testing_complete_result_set::resize_predicted_output_neuron_value_set(unsigned int entry_count)
	{
		predicted_output_neuron_value_set->neuron_value_list.resize(entry_count, std::vector<float>(actual_output_neuron_value_set->neuron_value_list.begin()->size()));
	}

	void testing_complete_result_set::recalculate_mse()
	{
		tr = testing_result_smart_ptr(new testing_result(ef));

		std::vector<std::vector<float> >::const_iterator it2 = predicted_output_neuron_value_set->neuron_value_list.begin();
		for(std::vector<std::vector<float> >::const_iterator it1 = actual_output_neuron_value_set->neuron_value_list.begin(); it1 != actual_output_neuron_value_set->neuron_value_list.end(); ++it1, ++it2)
		{
			const float * it_actual = &(*it1->begin());
			const float * it_predicted = &(*it2->begin());
			tr->add_error(tr->ef->calculate_error(it_actual, it_predicted, it1->size()));
		}
	}
}
