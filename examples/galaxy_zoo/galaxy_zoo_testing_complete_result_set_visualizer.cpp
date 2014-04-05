/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "galaxy_zoo_testing_complete_result_set_visualizer.h"

#include <boost/format.hpp>

galaxy_zoo_testing_complete_result_set_visualizer::galaxy_zoo_testing_complete_result_set_visualizer(nnforge::normalize_data_transformer_smart_ptr nds)
	: nds(nds)
{
}

galaxy_zoo_testing_complete_result_set_visualizer::~galaxy_zoo_testing_complete_result_set_visualizer()
{
}

void galaxy_zoo_testing_complete_result_set_visualizer::dump(
	std::ostream& out,
	const nnforge::testing_complete_result_set& val) const
{
	nnforge::testing_complete_result_set_visualizer::dump(out, val);

	unsigned int feature_map_count = nds->mul_add_list.size();
	unsigned int elem_count_per_feature_map = val.predicted_output_neuron_value_set->neuron_value_list[0].size() / feature_map_count;

	float sum = 0.0F;

	std::vector<std::vector<float> >::const_iterator predicted_it = val.predicted_output_neuron_value_set->neuron_value_list.begin();
	for(std::vector<std::vector<float> >::const_iterator actual_it = val.actual_output_neuron_value_set->neuron_value_list.begin();
		actual_it != val.actual_output_neuron_value_set->neuron_value_list.end();
		actual_it++, predicted_it++)
	{
		const std::vector<float>& predicted_values = *predicted_it;
		const std::vector<float>& actual_values = *actual_it;

		std::vector<float>::const_iterator predicted_it2 = predicted_values.begin();
		std::vector<float>::const_iterator actual_it2 = actual_values.begin();

		float local_sum = 0.0F;
		std::vector<std::pair<float, float> >::const_iterator mul_add_it = nds->mul_add_list.begin();
		for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id, ++mul_add_it)
		{
			float local_sum2 = 0.0F;

			for(int i = 0; i < elem_count_per_feature_map; ++i, ++predicted_it2, ++actual_it2)
			{
				float predicted_val = *predicted_it2 * mul_add_it->first + mul_add_it->second;
				float actual_val = *actual_it2 * mul_add_it->first + mul_add_it->second;
				float predicted_val_clipped = std::max(std::min(predicted_val, 1.0F), 0.0F);
				float dif = predicted_val_clipped - actual_val;
				local_sum2 += dif * dif;
			}

			local_sum += local_sum2;
		}

		sum += local_sum;
	}

	float rmse = sqrtf(sum / (static_cast<float>(val.predicted_output_neuron_value_set->neuron_value_list.size()) * static_cast<float>(val.predicted_output_neuron_value_set->neuron_value_list[0].size())));
//	float rmse = sqrtf(val.mse->get_mse() / static_cast<float>(val.mse->cumulative_mse_list.size()) * 2.0F);

	out << ", " << (boost::format("RMSE %|1$.3e|") % rmse).str();
}
