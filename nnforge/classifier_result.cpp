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

#include "classifier_result.h"

#include <algorithm>
#include <boost/format.hpp>

namespace nnforge
{
	classifier_result::classifier_result()
	{
	}

	classifier_result::classifier_result(
		const output_neuron_class_set& neuron_class_set_predicted,
		const output_neuron_class_set& neuron_class_set_actual)
		: predicted_and_actual_class_pair_id_list(neuron_class_set_predicted.class_id_list.size())
	{
		std::vector<std::pair<unsigned int, unsigned int> >::iterator dest_it = predicted_and_actual_class_pair_id_list.begin();
		std::vector<unsigned int>::const_iterator it2 = neuron_class_set_actual.class_id_list.begin();
		for(std::vector<unsigned int>::const_iterator it = neuron_class_set_predicted.class_id_list.begin(); it != neuron_class_set_predicted.class_id_list.end(); it++)
		{
			unsigned int predicted_class_id = *it;
			unsigned int actual_class_id = *it2;

			*dest_it = std::make_pair(predicted_class_id, actual_class_id);

			it2++;
			dest_it++;
		}
	}

	float classifier_result::get_invalid_ratio() const
	{
		unsigned int invalid_count = 0;

		for(std::vector<std::pair<unsigned int, unsigned int> >::const_iterator it = predicted_and_actual_class_pair_id_list.begin(); it != predicted_and_actual_class_pair_id_list.end(); it++)
		{
			if (it->first != it->second)
				invalid_count++;
		}

		return static_cast<float>(invalid_count) / static_cast<float>(predicted_and_actual_class_pair_id_list.size());
	}

	std::ostream& operator<< (std::ostream& out, const classifier_result& val)
	{
		out << (boost::format("Invalid ratio %|1$.2f|%%") % (val.get_invalid_ratio() * 100.0F)).str();

		return out;
	}
}
