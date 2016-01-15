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

#include "backward_propagation.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>
#include <boost/chrono.hpp>
#include <boost/filesystem/fstream.hpp>

namespace nnforge
{
	backward_propagation::backward_propagation(
		const network_schema& schema,
		const std::vector<std::string>& output_layer_names,
		const std::vector<std::string>& error_source_layer_names,
		const std::vector<std::string>& exclude_data_update_layer_names,
		debug_state::ptr debug)
		: output_layer_names(output_layer_names)
		, error_source_layer_names(error_source_layer_names)
		, exclude_data_update_layer_names(exclude_data_update_layer_names)
		, debug(debug)
	{
		if (error_source_layer_names.empty())
			throw neural_network_exception("No error source layers specified for backward_propagation");

		this->schema = network_schema::const_ptr(new network_schema(schema.get_required_layers(
			output_layer_names,
			error_source_layer_names,
			exclude_data_update_layer_names)));
		if (debug->is_debug())
		{
			boost::filesystem::ofstream out(debug->get_path_to_unique_file("backward_prop_schema_reduced", "gv"), std::ios_base::out | std::ios_base::trunc);
			this->schema->write_gv(out);
		}

		action_schema = this->schema->get_actions_for_backward_propagation(
			output_layer_names,
			error_source_layer_names,
			exclude_data_update_layer_names,
			same_output_action_sets);
		for(std::vector<std::vector<layer_name_with_action> >::const_iterator it = same_output_action_sets.begin(); it != same_output_action_sets.end(); ++it)
		{
			const std::vector<layer_name_with_action>& same_output_actions = *it;
			for(std::vector<layer_name_with_action>::const_iterator it2 = same_output_actions.begin(); it2 != same_output_actions.end() - 1; ++it2)
				add_output_actions.insert(*it2);
			if (debug->is_debug())
			{
				std::stringstream s;
				s << "Same output for actions: ";
				for(std::vector<layer_name_with_action>::const_iterator it2 = same_output_actions.begin(); it2 != same_output_actions.end(); ++it2)
				{
					if (it2 != same_output_actions.begin())
						s << ", ";
					s << it2->get_name() << " " << it2->get_action().str();
				}
				debug->output_message(s.str().c_str());
			}
		}
		if (debug->is_debug())
		{
			std::vector<layer_name_with_action> actions = action_schema->get_actions();
			std::map<layer_name_with_action, unsigned int> layer_name_with_action_color_map;
			for(std::vector<layer_name_with_action>::const_iterator it = actions.begin(); it != actions.end(); ++it)
			{
				unsigned int color_id;
				switch (it->get_action().get_action_type())
				{
				case layer_action::forward:
					color_id = 0;
					break;
				case layer_action::backward_data:
					color_id = 1;
					break;
				case layer_action::backward_weights:
					color_id = 2;
					break;
				case layer_action::update_weights:
					color_id = 3;
					break;
				default:
					color_id = 4;
					break;
				}
				layer_name_with_action_color_map.insert(std::make_pair(*it, color_id));
			}

			boost::filesystem::ofstream out(debug->get_path_to_unique_file("backward_prop_action_schema", "gv"), std::ios_base::out | std::ios_base::trunc);
			this->action_schema->write_gv(out, layer_name_with_action_color_map);
		}

		cumulative_tiling_factor_map = this->schema->get_cumulative_tiling_factor_map();
		output_layers_tiling_factor = 1;
		for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
		{
			if (it == output_layer_names.begin())
				output_layers_tiling_factor = cumulative_tiling_factor_map[*it];
			else if (output_layers_tiling_factor != cumulative_tiling_factor_map[*it])
				throw neural_network_exception((boost::format("Inconsistent tiling factors across output layers: %1% and %2%") % output_layers_tiling_factor % cumulative_tiling_factor_map[*it]).str());
		}

		std::vector<layer::const_ptr> data_layers = this->schema->get_data_layers();
		for(std::vector<layer::const_ptr>::const_iterator it = data_layers.begin(); it != data_layers.end(); ++it)
			data_layer_names.insert((*it)->instance_name);
	}

	backward_propagation::~backward_propagation()
	{
	}

	void backward_propagation::set_input_configuration_specific(const std::map<std::string, layer_configuration_specific>& input_configuration_specific_map)
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

		if (debug->is_debug())
		{
			boost::filesystem::ofstream out(debug->get_path_to_unique_file("backward_prop_schema_with_feature_map_configs", "gv"), std::ios_base::out | std::ios_base::trunc);
			this->schema->write_gv(out, layer_config_map, cumulative_tiling_factor_map);
		}

		update_flops();

		layer_config_map_modified();
	}

	void backward_propagation::update_flops()
	{
		flops = action_schema->get_flops(layer_config_map, cumulative_tiling_factor_map);
	}

	backward_propagation::stat backward_propagation::run(
		structured_data_bunch_reader& reader,
		structured_data_bunch_writer& writer,
		network_data& data,
		network_data::ptr momentum_data,
		network_data::ptr momentum_data2,
		const std::map<std::string, std::vector<float> >& learning_rates,
		unsigned int batch_size,
		float weight_decay,
		training_momentum momentum,
		unsigned int epoch_id)
	{
		backward_propagation::stat res;

		// Check data-schema consistency
		data.check_network_data_consistency(schema->get_layers());

		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		set_input_configuration_specific(reader.get_config_map());
		structured_data_bunch_reader::ptr narrow_reader = reader.get_narrow_reader(data_layer_names);
		res.flops_per_entry = flops;
		std::vector<std::string> data_layer_name_list(data_layer_names.begin(), data_layer_names.end());
		std::map<std::string, layer_configuration_specific> output_config_map;
		for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
			output_config_map[*it] = layer_config_map[*it];
		writer.set_config_map(output_config_map);
		std::pair<unsigned int, std::map<std::string, std::vector<float> > > p;
		if (narrow_reader)
			p = actual_run(
				*narrow_reader,
				writer,
				data,
				momentum_data,
				momentum_data2,
				learning_rates,
				batch_size,
				weight_decay,
				momentum,
				epoch_id);
		else
			p = actual_run(
				reader,
				writer,
				data,
				momentum_data,
				momentum_data2,
				learning_rates,
				batch_size,
				weight_decay,
				momentum,
				epoch_id);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
		res.entry_processed_count = p.first;
		res.average_absolute_updates = p.second;
		res.total_seconds = sec.count();

		return res;
	}

	std::ostream& operator<< (std::ostream& out, const backward_propagation::stat& val)
	{
		float gflops = val.flops_per_entry * static_cast<float>(val.entry_processed_count) / val.total_seconds * 1.0e-9F;
		out << (boost::format("%|1$.2f| seconds, %2% entries, %|3$.2e| flops per entry, %|4$.1f| GFLOPS") % val.total_seconds % val.entry_processed_count % val.flops_per_entry % gflops).str();
		return out;
	}
}
