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

#include "forward_propagation.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <boost/chrono.hpp>
#include <boost/filesystem/fstream.hpp>

namespace nnforge
{
	forward_propagation::forward_propagation(
		const network_schema& schema,
		const std::vector<std::string>& output_layer_names,
		debug_state::ptr debug)
		: output_layer_names(output_layer_names)
		, debug(debug)
	{
		if (output_layer_names.empty())
			throw neural_network_exception("No output layers specified for forward_propagation");

		this->schema = network_schema::const_ptr(new network_schema(schema.get_required_layers(output_layer_names)));
		if (debug->is_debug())
		{
			boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_schema_reduced", "gv"), std::ios_base::out | std::ios_base::trunc);
			this->schema->write_gv(out);
		}

		action_schema = this->schema->get_actions_for_forward_propagation(output_layer_names);
		if (debug->is_debug())
		{
			boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_action_schema", "gv"), std::ios_base::out | std::ios_base::trunc);
			this->action_schema->write_gv(out);
		}

		cumulative_tiling_factor_map = this->schema->get_cumulative_tiling_factor_map();

		std::vector<layer::const_ptr> data_layers = this->schema->get_data_layers();
		for(std::vector<layer::const_ptr>::const_iterator it = data_layers.begin(); it != data_layers.end(); ++it)
			data_layer_names.insert((*it)->instance_name);
	}

	forward_propagation::~forward_propagation()
	{
	}

	void forward_propagation::clear_data()
	{
		actual_clear_data();
	}

	void forward_propagation::set_data(const network_data& data)
	{
		std::vector<layer::const_ptr> layer_list = schema->get_layers();
		network_data::const_ptr new_data(new network_data(layer_list, data));
		new_data->check_network_data_consistency(layer_list);
		actual_set_data(new_data);
	}

	void forward_propagation::set_input_configuration_specific(const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map)
	{
		bool same_input_config = true;
		std::map<std::string, layer_configuration_specific> input_configuration_specific_map_filtered;
		for(std::map<std::string, layer_configuration_specific>::const_iterator it = input_configuration_specific_map.begin(); it != input_configuration_specific_map.end(); ++it)
		{
			if (data_layer_names.find(it->first) == data_layer_names.end())
				continue;

			input_configuration_specific_map_filtered.insert(*it);

			std::map<std::string, layer_configuration_specific>::const_iterator it2 = layer_config_map.find(it->first);
			if ((it2 == layer_config_map.end()) || (it->second != it2->second))
			{
				same_input_config = false;
			}
		}
		if (same_input_config)
			return;

		layer_config_map = schema->get_layer_configuration_specific_map(input_configuration_specific_map_filtered);

		update_flops();

		layer_config_map_modified();
	}

	void forward_propagation::update_flops()
	{
		flops = action_schema->get_flops(layer_config_map, cumulative_tiling_factor_map);
	}

	forward_propagation::stat forward_propagation::run(
		structured_data_bunch_reader& reader,
		structured_data_bunch_writer& writer)
	{
		forward_propagation::stat res;

		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		set_input_configuration_specific(reader.get_config_map());
		res.flops_per_entry = flops;
		std::vector<std::string> data_layer_name_list(data_layer_names.begin(), data_layer_names.end());
		std::map<std::string, layer_configuration_specific> output_config_map;
		for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
			output_config_map[*it] = layer_config_map[*it];
		writer.set_config_map(output_config_map);
		res.entry_processed_count = actual_run(reader, writer);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
		res.total_seconds = sec.count();

		return res;
	}

	std::ostream& operator<< (std::ostream& out, const forward_propagation::stat& val)
	{
		float gflops = val.flops_per_entry * static_cast<float>(val.entry_processed_count) / val.total_seconds * 1.0e-9F;
		out << (boost::format("%|1$.2f| seconds, %2% entries, %|3$.2e| flops per entry, %|4$.1f| GFLOPS") % val.total_seconds % val.entry_processed_count % val.flops_per_entry % gflops).str();
		return out;
	}
}
