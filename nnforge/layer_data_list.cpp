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

#include "layer_data_list.h"

#include "neural_network_exception.h"

#include <numeric>
#include <boost/format.hpp>

namespace nnforge
{
	layer_data_list::layer_data_list()
	{
	}

	layer_data_list::layer_data_list(const const_layer_list& layer_list, float val)
	{
		resize(layer_list.size());
		for(unsigned int i = 0; i < layer_list.size(); ++i)
		{
			at(i) = layer_list[i]->create_layer_data();
			at(i)->fill(val);
		}
	}

	void layer_data_list::check_consistency(const const_layer_list& layer_list) const
	{
		if (size() != layer_list.size())
			throw neural_network_exception("data count is not equal layer count");
		for(unsigned int i = 0; i < size(); ++i)
			layer_list[i]->check_layer_data_consistency(*at(i));
	}

	std::string layer_data_list::get_stat() const
	{
		std::string stat = "";

		unsigned int layer_id = 0;
		for(layer_data_list::const_iterator it = begin(); it != end(); ++it)
		{
			std::string layer_stat;

			for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
			{
				const std::vector<float>& data = *it2;

				double sum = std::accumulate(data.begin(), data.end(), 0.0);
				float avg = static_cast<float>(sum / data.size());

				if (!layer_stat.empty())
					layer_stat += ", ";
				layer_stat += (boost::format("%|1$.5e|") % avg).str();
			}

			if (!layer_stat.empty())
			{
				if (!stat.empty())
					stat += ", ";
				stat += (boost::format("Layer %1% - (%2%)") % layer_id % layer_stat).str();
			}

			layer_id++;
		}

		return stat;
	}

	void layer_data_list::fill(float val)
	{
		for(layer_data_list::iterator it = begin(); it != end(); ++it)
			(*it)->fill(val);
	}

	void layer_data_list::random_fill(
		float min,
		float max,
		random_generator& gen)
	{
		for(layer_data_list::iterator it = begin(); it != end(); ++it)
			(*it)->random_fill(min, max, gen);
	}
}
