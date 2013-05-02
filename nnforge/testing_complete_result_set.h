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

#pragma once

#include "testing_result.h"
#include "output_neuron_value_set.h"

namespace nnforge
{
	class testing_complete_result_set
	{
	protected:
		testing_complete_result_set();

	public:
		testing_complete_result_set(output_neuron_value_set_smart_ptr actual_output_neuron_value_set);

        void recalculate_mse();

		void resize_predicted_output_neuron_value_set(unsigned int entry_count);

		testing_result_smart_ptr mse;
		output_neuron_value_set_smart_ptr predicted_output_neuron_value_set;
		output_neuron_value_set_smart_ptr actual_output_neuron_value_set;
	};
}
