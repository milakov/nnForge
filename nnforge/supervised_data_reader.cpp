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

#include "supervised_data_reader.h"

#include "neural_network_exception.h"

#include <vector>
#include <limits>
#include <cmath>

namespace nnforge
{
	supervised_data_reader::supervised_data_reader()
	{
	}

	supervised_data_reader::~supervised_data_reader()
	{
	}

	output_neuron_value_set_smart_ptr supervised_data_reader::get_output_neuron_value_set(unsigned int sample_count)
	{
		reset();

		unsigned int entry_count = get_entry_count();
		unsigned int output_neuron_count = get_output_configuration().get_neuron_count();

		output_neuron_value_set_smart_ptr res(new output_neuron_value_set(entry_count, output_neuron_count));

		for(std::vector<std::vector<float> >::iterator it = res->neuron_value_list.begin(); it != res->neuron_value_list.end(); it++)
		{
			std::vector<float>& output_neurons = *it;

			read(0, &(*output_neurons.begin()));
		}

		res->compact(sample_count);

		return res;
	}

	bool supervised_data_reader::read(void * input_elems)
	{
		return read(input_elems, 0);
	}

	std::vector<feature_map_data_stat> supervised_data_reader::get_feature_map_output_data_stat_list()
	{
		std::vector<feature_map_data_stat> res;

		reset();

		unsigned int entry_count = get_entry_count();
		if (entry_count == 0)
			throw neural_network_exception("Unable to stat data reader with no entries");

		layer_configuration_specific output_configuration = get_output_configuration();
		res.resize(output_configuration.feature_map_count);

		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
		{
			it->min = std::numeric_limits<float>::max();
			it->max = -std::numeric_limits<float>::max();
			it->average = 0.0F;
			it->std_dev = 0.0F;
		}

		std::vector<float> output_data(output_configuration.get_neuron_count());
		unsigned int neuron_count_per_feature_map = output_configuration.get_neuron_count_per_feature_map();

		while(read(0, &(*output_data.begin())))
		{
			std::vector<float>::const_iterator data_it = output_data.begin();
			for(std::vector<feature_map_data_stat>::iterator fm_it = res.begin(); fm_it != res.end(); ++fm_it)
			{
				float current_sum = 0.0F;
				for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				{
					float val = *data_it;
					fm_it->min = std::min(fm_it->min, val);
					fm_it->max = std::max(fm_it->max, val);
					current_sum += val;
					++data_it;
				}
				fm_it->average += current_sum;
			}
		}

		float mult = 1.0F / ((float)entry_count * (float)neuron_count_per_feature_map);
		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
			it->average *= mult;

		reset();

		while(read(0, &(*output_data.begin())))
		{
			std::vector<float>::const_iterator data_it = output_data.begin();
			for(std::vector<feature_map_data_stat>::iterator fm_it = res.begin(); fm_it != res.end(); ++fm_it)
			{
				float current_sum = 0.0F;
				float average = fm_it->average;
				for(unsigned int i = 0; i < neuron_count_per_feature_map; ++i)
				{
					float val = *data_it;
					float diff = val - average;
					current_sum += diff * diff;
					++data_it;
				}
				fm_it->std_dev += current_sum;
			}
		}

		for(std::vector<feature_map_data_stat>::iterator it = res.begin(); it != res.end(); ++it)
			it->std_dev = sqrtf(it->std_dev * mult);

		return res;
	}

	void supervised_data_reader::fill_class_buckets_entry_id_lists(std::vector<randomized_classifier_keeper>& class_buckets_entry_id_lists)
	{
		unsigned int output_neuron_count = get_output_configuration().get_neuron_count();

		class_buckets_entry_id_lists.resize(output_neuron_count + 1);

		std::vector<float> output(output_neuron_count);

		unsigned int entry_id = 0;
		while (read(0, &(*output.begin())))
		{
			float min_value = *std::min_element(output.begin(), output.end());
			std::vector<float>::iterator max_elem = std::max_element(output.begin(), output.end());

			if ((min_value < *max_elem) || (*max_elem > 0.0F))
				class_buckets_entry_id_lists[max_elem - output.begin()].push(entry_id);
			else
				class_buckets_entry_id_lists[output.size()].push(entry_id);

			++entry_id;
		}
	}

	randomized_classifier_keeper::randomized_classifier_keeper()
		: pushed_count(0)
		, remaining_ratio(0.0F)
	{
	}

	bool randomized_classifier_keeper::is_empty()
	{
		return entry_id_list.empty();
	}

	float randomized_classifier_keeper::get_ratio()
	{
		return remaining_ratio;
	}

	void randomized_classifier_keeper::push(unsigned int entry_id)
	{
		entry_id_list.push_back(entry_id);
		++pushed_count;

		update_ratio();
	}

	unsigned int randomized_classifier_keeper::peek_random(random_generator& rnd)
	{
		nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(entry_id_list.size()) - 1);

		unsigned int index = dist(rnd);
		unsigned int entry_id = entry_id_list[index];

		unsigned int leftover_entry_id = entry_id_list[entry_id_list.size() - 1];
		entry_id_list[index] = leftover_entry_id;

		entry_id_list.pop_back();

		update_ratio();

		return entry_id;
	}

	void randomized_classifier_keeper::update_ratio()
	{
		remaining_ratio = pushed_count > 0 ? static_cast<float>(entry_id_list.size()) / static_cast<float>(pushed_count) : 0.0F;
	}
}
