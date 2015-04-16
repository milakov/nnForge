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

#include <vector>

#include "output_neuron_value_set.h"
#include "nn_types.h"

namespace nnforge
{
	class output_neuron_class_set
	{
	public:
		output_neuron_class_set(unsigned int top_n);

		output_neuron_class_set(
			const output_neuron_value_set& neuron_value_set,
			unsigned int top_n,
			float threshold = 0.5F); // Used for binary classifiers

		std::vector<unsigned int> class_id_list;

		unsigned int top_n;
	};

	typedef nnforge_shared_ptr<output_neuron_class_set> output_neuron_class_set_smart_ptr;
}
