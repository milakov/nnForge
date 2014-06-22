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

#include "output_neuron_class_set.h"

#include "neural_network_exception.h"

#include <algorithm>
#include <map>
#include <limits>
#include <boost/format.hpp>

namespace nnforge
{
	output_neuron_class_set::output_neuron_class_set(unsigned int top_n)
		: top_n(top_n)
	{
	}

	output_neuron_class_set::output_neuron_class_set(const output_neuron_value_set& neuron_value_set, unsigned int top_n)
		: top_n(top_n)
		, class_id_list(neuron_value_set.neuron_value_list.size() * top_n)
	{
		if (neuron_value_set.neuron_value_list.empty())
			throw neural_network_exception("Empty neuron_value_set passed to output_neuron_class_set");
		unsigned int class_count = neuron_value_set.neuron_value_list.front().size();
		if (class_count > 1)
		{
			if (class_count < top_n)
				throw neural_network_exception((boost::format("Class count is %1%, it smaller than top %2% requested from output_neuron_class_set") % class_count % top_n).str());

			std::vector<unsigned int>::iterator dest_it = class_id_list.begin();
			std::vector<std::pair<float, unsigned int> > current_best_elems(top_n);
			for(std::vector<std::vector<float> >::const_iterator it = neuron_value_set.neuron_value_list.begin(); it != neuron_value_set.neuron_value_list.end(); it++)
			{
				const std::vector<float>& neuron_values = *it;
				std::fill_n(current_best_elems.begin(), top_n, std::make_pair(-std::numeric_limits<float>::max(), class_count));

				for(int class_id = 0; class_id < class_count; ++class_id)
				{
					float neuron_val = neuron_values[class_id];
					if (neuron_val <= current_best_elems.back().first)
						continue;

					unsigned int target_position = top_n - 1;
					for(;target_position >= 1; --target_position)
					{
						if (current_best_elems[target_position - 1].first >= neuron_val)
							break;

						current_best_elems[target_position] = current_best_elems[target_position - 1];
					}
					current_best_elems[target_position] = std::make_pair(neuron_val, class_id);
				}

				for(int i = 0; i < top_n; ++i)
				{
					*dest_it = current_best_elems[i].second;
					++dest_it;
				}
			}
		} // if (class_count > 1)
		else
		{
			class_count = 2;
			if (class_count < top_n)
				throw neural_network_exception((boost::format("Class count is %1%, it smaller than top %2% requested from output_neuron_class_set") % class_count % top_n).str());

			std::vector<unsigned int>::iterator dest_it = class_id_list.begin();
			std::vector<unsigned int> best_elems(2);
			for(std::vector<std::vector<float> >::const_iterator it = neuron_value_set.neuron_value_list.begin(); it != neuron_value_set.neuron_value_list.end(); it++)
			{
				const std::vector<float>& neuron_values = *it;
				float val = neuron_values.front();
				if (val >= 0.5F)
				{
					best_elems[0] = 1;
					best_elems[1] = 0;
				}
				else
				{
					best_elems[0] = 0;
					best_elems[1] = 1;
				}

				for(int i = 0; i < top_n; ++i)
				{
					*dest_it = best_elems[i];
					++dest_it;
				}
			}
		}
	}
}
