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

#include "layer_configuration_specific.h"
#include "output_neuron_value_set.h"

#include <vector>
#include <memory>
#include <numeric>
#include <math.h>

namespace nnforge
{
	template <typename input_data_type> class supervised_data_reader
	{
	public:

		virtual ~supervised_data_reader()
		{
		}

		// The method should return true in case entry is read and false if there is no more entries available (and no entry is read in this case)
		// If any parameter is null the method should just discard corresponding data
		virtual bool read(
			input_data_type * input_elems,
			float * output_elems) = 0;

		virtual void reset() = 0;

		virtual layer_configuration_specific get_input_configuration() const = 0;

		virtual layer_configuration_specific get_output_configuration() const = 0;

		virtual unsigned int get_entry_count() const = 0;

		std::vector<float> get_feature_map_average()
		{
			reset();

			unsigned int feature_map_count = get_input_configuration().feature_map_count;
			unsigned int neuron_count_per_feature_map = get_input_configuration().get_neuron_count_per_feature_map();
			std::vector<float> res(feature_map_count, 0.0F);
			std::vector<input_data_type> inp(get_input_configuration().get_neuron_count());
			while (read(&(*inp.begin()), 0))
			{
				for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
				{
					float sum_local = 0.0F;
					for(typename std::vector<input_data_type>::const_iterator it = inp.begin() + feature_map_id * neuron_count_per_feature_map;
						it != inp.begin() + (feature_map_id + 1) * neuron_count_per_feature_map;
						++it)
					{
						sum_local += (static_cast<float>(*it) * (1.0F / 255.0F));
					}
					res[feature_map_id] += sum_local;
				}
			}

			float mult = 1.0F / static_cast<float>(get_entry_count() * neuron_count_per_feature_map);
			for(std::vector<float>::iterator it = res.begin(); it != res.end(); ++it)
				*it *= mult;

			return res;
		}

		std::vector<std::pair<float, float> > get_feature_map_min_max()
		{
			reset();

			unsigned int feature_map_count = get_input_configuration().feature_map_count;
			unsigned int neuron_count_per_feature_map = get_input_configuration().get_neuron_count_per_feature_map();
			std::vector<std::pair<float, float> > res(feature_map_count, std::make_pair<float, float>(1.0e37F, -1.0e37F));
			std::vector<input_data_type> inp(get_input_configuration().get_neuron_count());
			while (read(&(*inp.begin()), 0))
			{
				for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
				{
					float min_local = 1.0e37F;
					float max_local = -1.0e37F;
					for(typename std::vector<input_data_type>::const_iterator it = inp.begin() + feature_map_id * neuron_count_per_feature_map;
						it != inp.begin() + (feature_map_id + 1) * neuron_count_per_feature_map;
						++it)
					{
						float val = static_cast<float>(*it) * (1.0F / 255.0F);
						min_local = std::min<float>(min_local, val);
						max_local = std::max<float>(max_local, val);
					}
					res[feature_map_id].first = std::min<float>(res[feature_map_id].first, min_local);
					res[feature_map_id].second = std::max<float>(res[feature_map_id].second, max_local);
				}
			}

			return res;
		}

		std::vector<float> get_feature_map_std_dev(const std::vector<float>& avg)
		{
			reset();

			unsigned int feature_map_count = get_input_configuration().feature_map_count;
			unsigned int neuron_count_per_feature_map = get_input_configuration().get_neuron_count_per_feature_map();
			std::vector<float> res(feature_map_count, 0.0F);
			std::vector<input_data_type> inp(get_input_configuration().get_neuron_count());
			while (read(&(*inp.begin()), 0))
			{
				for(unsigned int feature_map_id = 0; feature_map_id < feature_map_count; ++feature_map_id)
				{
					float sum_local = 0.0F;
					float current_avg = avg[feature_map_id];
					for(typename std::vector<input_data_type>::const_iterator it = inp.begin() + feature_map_id * neuron_count_per_feature_map;
						it != inp.begin() + (feature_map_id + 1) * neuron_count_per_feature_map;
						++it)
					{
						float val = (static_cast<float>(*it) * (1.0F / 255.0F));
						float diff = val - current_avg;
                        sum_local += diff * diff;
					}
					res[feature_map_id] += sum_local;
				}
			}

			float mult = 1.0F / static_cast<float>(get_entry_count() * neuron_count_per_feature_map);
			for(std::vector<float>::iterator it = res.begin(); it != res.end(); ++it)
				*it = sqrtf(*it * mult);

			return res;
		}

		output_neuron_value_set_smart_ptr get_output_neuron_value_set()
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

			return res;
		}

	protected:
		supervised_data_reader()
		{
		}

	private:
		supervised_data_reader(const supervised_data_reader&);
		supervised_data_reader& operator =(const supervised_data_reader&);
	};

	typedef supervised_data_reader<unsigned char> supervised_data_reader_byte;
	typedef supervised_data_reader<float> supervised_data_reader_float;

	typedef std::tr1::shared_ptr<supervised_data_reader_byte> supervised_data_reader_byte_smart_ptr;
}
