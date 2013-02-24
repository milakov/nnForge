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

#include "hyperbolic_tangent_layer.h"
#include "output_neuron_value_set.h"

#include <vector>

namespace nnforge
{
	class roc_result
	{
	public:
		roc_result(
			const output_neuron_value_set& predicted_value_set,
			const output_neuron_value_set& actual_value_set,
			unsigned int segment_count = 1000,
			float min_val = - hyperbolic_tangent_layer::major_multiplier,
			float max_val = hyperbolic_tangent_layer::major_multiplier);

		float get_accuracy(float threshold) const;

		unsigned int segment_count;

		std::vector<unsigned int> values_for_positive_elems;
		std::vector<unsigned int> values_for_negative_elems;

		float min_val;
		float max_val;

		unsigned int actual_positive_elem_count;
		unsigned int actual_negative_elem_count;
	};

	std::ostream& operator<< (std::ostream& out, const roc_result& val);

	typedef std::tr1::shared_ptr<roc_result> roc_result_smart_ptr;
}
