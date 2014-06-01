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

#include <vector>
#include <ostream>

#include "nn_types.h"
#include "output_neuron_class_set.h"

namespace nnforge
{
	class classifier_result
	{
	public:
		classifier_result();

		classifier_result(
			const output_neuron_class_set& neuron_class_set_predicted,
			const output_neuron_class_set& neuron_class_set_actual);

		std::vector<float> get_invalid_ratio_list() const;

		std::vector<unsigned int> predicted_class_id_list;
		std::vector<unsigned int> actual_class_id_list;
		unsigned int top_n;
	};

	std::ostream& operator<< (std::ostream& out, const classifier_result& val);

	typedef nnforge_shared_ptr<classifier_result> classifier_result_smart_ptr;
}
