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

#include "neural_network_exception.h"

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
		: top_n(neuron_class_set_predicted.top_n)
	{
		if (neuron_class_set_actual.top_n != 1)
			throw neural_network_exception((boost::format("classifier_result is not implemented for top_n of actual classes equal %1%") % top_n).str());

		predicted_class_id_list = neuron_class_set_predicted.class_id_list;
		actual_class_id_list = neuron_class_set_actual.class_id_list;
	}

	std::vector<float> classifier_result::get_invalid_ratio_list() const
	{
		std::vector<unsigned int> invalid_counts(top_n, 0);

		std::vector<unsigned int>::const_iterator predicted_it = predicted_class_id_list.begin();
		for(std::vector<unsigned int>::const_iterator it = actual_class_id_list.begin(); it != actual_class_id_list.end(); ++it, predicted_it += top_n)
		{
			unsigned int actual_class_id = *it;

			for(unsigned int i = 0; i < top_n; ++i)
			{
				if (actual_class_id == *(predicted_it + i))
					break;

				invalid_counts[i]++;
			}
		}

		std::vector<float> invalid_ratios(top_n);
		for(unsigned int i = 0; i < top_n; ++i)
			invalid_ratios[i] = static_cast<float>(invalid_counts[i]) / static_cast<float>(actual_class_id_list.size());

		return invalid_ratios;
	}

	std::ostream& operator<< (std::ostream& out, const classifier_result& val)
	{
		std::vector<float> invalid_ratios = val.get_invalid_ratio_list();
		out << "Error rate";
		for(int i = 0; i < invalid_ratios.size(); ++i)
		{
			if (i > 0)
				out << ",";
			out << (boost::format(" Top-%1% %|2$.2f|%%") % (i + 1) % (invalid_ratios[i] * 100.0F)).str();
		}

		return out;
	}
}
