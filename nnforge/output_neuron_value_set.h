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
#include <memory>

namespace nnforge
{
	class output_neuron_value_set
	{
	public:
		enum merge_type_enum
		{
			merge_average,
			merge_median
		};

		output_neuron_value_set();

		output_neuron_value_set(
			unsigned int entry_count,
			unsigned int neuron_count);

		output_neuron_value_set(
			const std::vector<std::tr1::shared_ptr<output_neuron_value_set> >& source_output_neuron_value_set_list,
			merge_type_enum merge_type);

		void clamp(
			float min_val,
			float max_val);

		void compact(unsigned int sample_count);

		std::vector<std::vector<float> > neuron_value_list;
	};

	typedef std::tr1::shared_ptr<output_neuron_value_set> output_neuron_value_set_smart_ptr;
}
