/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "structured_data_bunch_mix_reader.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	structured_data_bunch_mix_reader::structured_data_bunch_mix_reader(
		structured_data_bunch_reader::ptr main_reader,
		structured_data_bunch_reader::ptr auxiliary_reader,
		float auxiliary_reader_part)
		: main_reader(main_reader)
		, auxiliary_reader(auxiliary_reader)
		, auxiliary_reader_part(auxiliary_reader_part)
	{
		update_redirect_entry_list();
	}

	structured_data_bunch_mix_reader::~structured_data_bunch_mix_reader()
	{
	}

	void structured_data_bunch_mix_reader::update_redirect_entry_list()
	{
		redirect_entry_list.clear();

		int main_entry_count = main_reader->get_entry_count();
		if (main_entry_count < 0)
			throw neural_network_exception("structured_data_bunch_mix_reader cannot function with unknown main_reader entry_count");

		int auxiliary_entry_count = auxiliary_reader->get_entry_count();
		if (auxiliary_entry_count < 0)
			throw neural_network_exception("structured_data_bunch_mix_reader cannot function with unknown auxiliary_reader entry_count");
		if (auxiliary_entry_count == 0)
			throw neural_network_exception("structured_data_bunch_mix_reader cannot function with zero auxiliary_reader entry_count");

		double auxiliary_reader_part_d = static_cast<double>(auxiliary_reader_part);

		int total_approximate_entry_count = static_cast<unsigned int>(static_cast<double>(main_entry_count) / (1.0 - auxiliary_reader_part_d)) + 1;
		redirect_entry_list.reserve(total_approximate_entry_count);

		int main_entry_count_redirected = 0;
		int auxiliary_entry_count_redirected = 0;

		while (true)
		{
			double mult = 1.0 / static_cast<double>(main_entry_count_redirected + auxiliary_entry_count_redirected + 1);
			double main_error = static_cast<double>(auxiliary_entry_count_redirected) * mult - auxiliary_reader_part_d;
			double auxiliary_error = main_error + mult;
			if (abs(auxiliary_error) < abs(main_error))
			{
				redirect_entry_list.push_back(-(auxiliary_entry_count_redirected % auxiliary_entry_count) - 1);
				++auxiliary_entry_count_redirected;
			}
			else
			{
				if (main_entry_count_redirected >= main_entry_count)
					break;

				redirect_entry_list.push_back(main_entry_count_redirected);
				++main_entry_count_redirected;
			}
		}
	}

	void structured_data_bunch_mix_reader::set_epoch(unsigned int epoch_id)
	{
		main_reader->set_epoch(epoch_id);
		auxiliary_reader->set_epoch(epoch_id);
		update_redirect_entry_list();
	}

	bool structured_data_bunch_mix_reader::read(
		unsigned int entry_id,
		const std::map<std::string, float *>& data_map)
	{
		if (entry_id >= static_cast<unsigned int>(redirect_entry_list.size()))
			return false;

		int redirected_entry_id = redirect_entry_list[entry_id];
		if (redirected_entry_id >= 0)
			return main_reader->read(redirected_entry_id, data_map);
		else
			return auxiliary_reader->read(-(redirected_entry_id + 1), data_map);
	}

	int structured_data_bunch_mix_reader::get_entry_count() const
	{
		return static_cast<int>(redirect_entry_list.size());
	}

	std::map<std::string, layer_configuration_specific> structured_data_bunch_mix_reader::get_config_map() const
	{
		std::map<std::string, layer_configuration_specific> main_res = main_reader->get_config_map();
		std::map<std::string, layer_configuration_specific> auxiliary_res = auxiliary_reader->get_config_map();

		std::map<std::string, layer_configuration_specific> total_res;
		for(std::map<std::string, layer_configuration_specific>::const_iterator main_it = main_res.begin(); main_it != main_res.end(); ++main_it)
		{
			std::map<std::string, layer_configuration_specific>::const_iterator auxiliary_it = auxiliary_res.find(main_it->first);
			if (auxiliary_it != auxiliary_res.end())
			{
				if (main_it->second == auxiliary_it->second)
					total_res.insert(*main_it);
				else
					throw neural_network_exception((boost::format("layer config mismatch for main and auxiliary readers for layer %1%") % main_it->first).str());
			}
		}

		return total_res;
	}

	structured_data_bunch_reader::ptr structured_data_bunch_mix_reader::get_narrow_reader(const std::set<std::string>& layer_names) const
	{
		structured_data_bunch_reader::ptr new_main_reader = main_reader->get_narrow_reader(layer_names);
		structured_data_bunch_reader::ptr new_auxiliary_reader = auxiliary_reader->get_narrow_reader(layer_names);

		if ((!new_main_reader) && (!new_auxiliary_reader))
			return structured_data_bunch_reader::ptr();

		if (!new_main_reader)
			new_main_reader = main_reader;
		if (!new_auxiliary_reader)
			new_auxiliary_reader = auxiliary_reader;

		return structured_data_bunch_reader::ptr(new structured_data_bunch_mix_reader(new_main_reader, new_auxiliary_reader, auxiliary_reader_part));
	}
}
