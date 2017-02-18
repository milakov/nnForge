/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "structured_data_bunch_writer.h"
#include "feature_map_data_stat.h"

#include <map>
#include <limits>

namespace nnforge
{
	class stat_data_bunch_writer : public structured_data_bunch_writer
	{
	public:
		typedef std::shared_ptr<stat_data_bunch_writer> ptr;

		stat_data_bunch_writer();

		virtual ~stat_data_bunch_writer() = default;

		virtual void set_config_map(const std::map<std::string, layer_configuration_specific> config_map);

		virtual void write(
			unsigned int entry_id,
			const std::map<std::string, const float *>& data_map);

		std::map<std::string, std::vector<feature_map_data_stat> > get_stat() const;

	private:
		struct running_stat
		{
		public:
			running_stat()
				: mean(0.0)
				, m2(0.0)
				, n(0.0)
				, min_val(std::numeric_limits<float>::max())
				, max_val(-std::numeric_limits<float>::max())
			{
			}

			double mean;
			double m2;
			double n;
			float min_val;
			float max_val;
		};

		std::map<std::string, std::vector<running_stat> > layer_name_to_running_stat_list_map;
		std::map<std::string, unsigned int> layer_name_to_neuron_count_per_feature_map_map;
		unsigned int entry_count;
	};
}
