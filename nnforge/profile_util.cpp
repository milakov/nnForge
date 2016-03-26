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

#include "profile_util.h"

#include <boost/format.hpp>
#include <algorithm>
#include <boost/filesystem/fstream.hpp>

namespace nnforge
{
	bool profile_util::compare_entry(const entry& i, const entry& j)
	{
		return i.seconds > j.seconds;
	}

	void profile_util::dump_layer_action_performance(
		profile_state::ptr profile,
		float max_flops,
		const char * action_prefix,
		unsigned int entry_count,
		const std::map<layer_name_with_action, float>& action_flops_per_entry,
		const std::map<layer_name_with_action, float>& action_seconds,
		const std::map<std::string, std::string>& layer_name_to_layer_type_map,
		float total_second)
	{
		{
			std::vector<entry> entries;
			for(std::map<layer_name_with_action, float>::const_iterator it = action_seconds.begin(); it != action_seconds.end(); ++it)
				entries.push_back(entry(it->first, it->second));
			std::sort(entries.begin(), entries.end(), compare_entry);

			boost::filesystem::path profile_path = profile->get_path_to_unique_file((boost::format("%1%_perf_per_layer_action") % action_prefix).str().c_str(), "csv");
			boost::filesystem::ofstream out(profile_path, std::ios_base::out | std::ios_base::trunc);
			out << "Layer\tLayer type\tAction\tTime of total\tRelative to peak perf\tAbsolute time, seconds\tAbsolute perf, GFLOPS" << std::endl;
			float max_gflops = max_flops * 1.0e-9F;
			for(std::vector<entry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
			{
				float gflops = action_flops_per_entry.find(it->action)->second * static_cast<float>(entry_count) / it->seconds * 1.0e-9F;
				float relative_gflops = gflops / max_gflops;

				out << it->action.get_name();
				out << "\t" << layer_name_to_layer_type_map.find(it->action.get_name())->second;
				out << "\t" << it->action.get_action().str();
				out << "\t" << (boost::format("%|1$.2f|%%") % (it->seconds / static_cast<float>(total_second) * 100.0F)).str();
				if (it->action.get_action().get_action_type() != layer_action::update_weights)
					out << "\t" << (boost::format("%|1$.2f|%%") % (relative_gflops * 100.0F)).str();
				else
					out << "\tNA";
				out << "\t" << it->seconds;
				if (it->action.get_action().get_action_type() != layer_action::update_weights)
					out << "\t" << gflops;
				else
					out << "\tNA";
				out << std::endl;
			}
		}

		{
			std::map<layer_name_with_action, double> action_flops_per_entry2;
			for(std::map<layer_name_with_action, float>::const_iterator it = action_flops_per_entry.begin(); it != action_flops_per_entry.end(); ++it)
				action_flops_per_entry2.insert(std::make_pair(layer_name_with_action(layer_name_to_layer_type_map.find(it->first.get_name())->second, it->first.get_action()), 0.0)).first->second += static_cast<double>(it->second);

			std::map<layer_name_with_action, double> action_seconds2;
			for(std::map<layer_name_with_action, float>::const_iterator it = action_seconds.begin(); it != action_seconds.end(); ++it)
				action_seconds2.insert(std::make_pair(layer_name_with_action(layer_name_to_layer_type_map.find(it->first.get_name())->second, it->first.get_action()), 0.0)).first->second += static_cast<double>(it->second);

			std::vector<entry> entries;
			for(std::map<layer_name_with_action, double>::const_iterator it = action_seconds2.begin(); it != action_seconds2.end(); ++it)
				entries.push_back(entry(it->first, static_cast<float>(it->second)));
			std::sort(entries.begin(), entries.end(), compare_entry);

			boost::filesystem::path profile_path = profile->get_path_to_unique_file((boost::format("%1%_perf_per_layer_type_action") % action_prefix).str().c_str(), "csv");
			boost::filesystem::ofstream out(profile_path, std::ios_base::out | std::ios_base::trunc);
			out << "Layer type\tAction\tTime of total\tRelative to peak perf\tAbsolute time, seconds\tAbsolute perf, GFLOPS" << std::endl;
			float max_gflops = max_flops * 1.0e-9F;
			for(std::vector<entry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
			{
				float gflops = static_cast<float>(action_flops_per_entry2.find(it->action)->second) * static_cast<float>(entry_count) / it->seconds * 1.0e-9F;
				float relative_gflops = gflops / max_gflops;

				out << it->action.get_name();
				out << "\t" << it->action.get_action().str();
				out << "\t" << (boost::format("%|1$.2f|%%") % (it->seconds / static_cast<float>(total_second) * 100.0F)).str();
				if (it->action.get_action().get_action_type() != layer_action::update_weights)
					out << "\t" << (boost::format("%|1$.2f|%%") % (relative_gflops * 100.0F)).str();
				else
					out << "\tNA";
				out << "\t" << it->seconds;
				if (it->action.get_action().get_action_type() != layer_action::update_weights)
					out << "\t" << gflops;
				else
					out << "\tNA";
				out << std::endl;
			}
		}
	}
}
