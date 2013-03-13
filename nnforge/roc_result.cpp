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

#include "roc_result.h"

#include <boost/format.hpp>
#include <algorithm>
#include <numeric>

namespace nnforge
{
	roc_result::roc_result(
		const output_neuron_value_set& predicted_value_set,
		const output_neuron_value_set& actual_value_set,
		unsigned int segment_count,
		float min_val,
		float max_val)
		: segment_count(segment_count)
		, min_val(min_val)
		, max_val(max_val)
		, actual_positive_elem_count(0)
		, actual_negative_elem_count(0)
		, values_for_positive_elems(segment_count)
		, values_for_negative_elems(segment_count)
	{
		float mult = 1.0F / (max_val - min_val);
		float segment_count_f = static_cast<float>(segment_count);
		std::vector<std::vector<float> >::const_iterator predicted_it = predicted_value_set.neuron_value_list.begin();
		for(std::vector<std::vector<float> >::const_iterator actual_it = actual_value_set.neuron_value_list.begin();
			actual_it != actual_value_set.neuron_value_list.end();
			actual_it++)
		{
			float actual_value = (*actual_it)[0];
			float predicted_value = (*predicted_it)[0];

			unsigned int bucket_id = std::min<unsigned int>(static_cast<unsigned int>(std::max<float>(std::min<float>((predicted_value - min_val) * mult, 1.0F), 0.0F) * segment_count_f), (segment_count - 1));

			if (actual_value > 0.0F)
			{
				values_for_positive_elems[bucket_id]++;
				actual_positive_elem_count++;
			}
			else
			{
				values_for_negative_elems[bucket_id]++;
				actual_negative_elem_count++;
			}

			predicted_it++;
		}
	}

	float roc_result::get_accuracy(float threshold) const
	{
		unsigned int starting_segment_id = static_cast<unsigned int>(std::max(std::min((threshold - min_val) / (max_val - min_val), 1.0F), 0.0F) * static_cast<float>(segment_count));

		unsigned int true_positive = std::accumulate(values_for_positive_elems.begin() + starting_segment_id, values_for_positive_elems.end(), 0);
		unsigned int true_negative = std::accumulate(values_for_negative_elems.begin(), values_for_negative_elems.begin() + starting_segment_id, 0);

		return static_cast<float>(true_positive + true_negative) / static_cast<float>(actual_positive_elem_count + actual_negative_elem_count);
	}

	float roc_result::get_auc() const
	{
		std::vector<float> true_positive_rates;
		{
			unsigned int current_positive_elems_count = 0;
			float mult = 1.0F / static_cast<float>(actual_positive_elem_count);
			for(std::vector<unsigned int>::const_reverse_iterator it = values_for_positive_elems.rbegin(); it != values_for_positive_elems.rend(); ++it)
			{
				current_positive_elems_count += *it;
				true_positive_rates.push_back(mult * static_cast<float>(current_positive_elems_count));
			}
		}
		true_positive_rates.push_back(1.0F);

		std::vector<float> false_positive_rates;
		{
			unsigned int current_negative_elems_count = 0;
			float mult = 1.0F / static_cast<float>(actual_negative_elem_count);
			for(std::vector<unsigned int>::const_reverse_iterator it = values_for_negative_elems.rbegin(); it != values_for_negative_elems.rend(); ++it)
			{
				current_negative_elems_count += *it;
				false_positive_rates.push_back(mult * static_cast<float>(current_negative_elems_count));
			}
		}
		false_positive_rates.push_back(1.0F);

		float sum = 0.0F;
		float previous_fpr = 0.0F;
		float previous_tpr = 0.0F;
		std::vector<float>::const_iterator tpr_it = true_positive_rates.begin();
		for(std::vector<float>::const_iterator fpr_it = false_positive_rates.begin(); fpr_it != false_positive_rates.end(); ++fpr_it, ++tpr_it)
		{
			float current_fpr = *fpr_it;
			float current_tpr = *tpr_it;

			if (current_fpr != previous_fpr)
				sum += (current_fpr - previous_fpr) * (previous_tpr + current_tpr) * 0.5F;

			previous_fpr = current_fpr;
			previous_tpr = current_tpr;
		}

		return sum;
	}

	std::ostream& operator<< (std::ostream& out, const roc_result& val)
	{
		out << (boost::format("AUC %|1$.5f|, accuracy at middle point %|2$.2f|%%") % val.get_auc() % (val.get_accuracy(0.0F) * 100.0F)).str();

		return out;
	}
}
