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

#include "forward_propagation_plain.h"

#include "layer_tester_plain_factory.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/fstream.hpp>

#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		const unsigned int forward_propagation_plain::max_max_entry_count = 1024;

		forward_propagation_plain::forward_propagation_plain(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug,
			plain_running_configuration::const_ptr plain_config)
			: forward_propagation(schema, output_layer_names, debug)
			, plain_config(plain_config)
			, max_entry_count(0)
			, temporary_working_fixed_size(0)
		{
			actions_in_execution_order = action_schema->get_actions_in_execution_order();

			// CPU is an easy to saturate device, we run everything in a single stream/thread, this will save some (maybe significant amount of) RAM
			network_action_schema::ptr sequential_action_schema(new network_action_schema());
			{
				std::vector<layer_name_with_action> dependencies;
				for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
				{
					sequential_action_schema->add_action(
						this->schema->get_layer(it->get_name()),
						it->get_action(),
						dependencies);
					dependencies.clear();
					dependencies.push_back(*it);
				}
			}
			action_schema = sequential_action_schema;

			if (debug->is_debug())
			{
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_plain_action_schema_sequential", "gv"), std::ios_base::out | std::ios_base::trunc);
				action_schema->write_gv(out);
			}

			for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
				testers.insert(
					std::make_pair(
						it->get_name(),
						layer_tester_plain_factory::singleton::get_const_instance().get_tester_plain_layer(this->schema->get_layer(it->get_name())->get_type_name())));
		}

		forward_propagation_plain::~forward_propagation_plain()
		{
		}

		void forward_propagation_plain::actual_set_data(network_data::const_ptr data)
		{
			net_data = data;
		}

		void forward_propagation_plain::actual_clear_data()
		{
			net_data.reset();
		}

		unsigned int forward_propagation_plain::actual_run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer)
		{
			unsigned int current_max_entry_count = max_entry_count;
			int reader_entry_count = reader.get_entry_count();
			if (reader_entry_count > 0)
				current_max_entry_count = std::min(current_max_entry_count, static_cast<unsigned int>(reader_entry_count));
			current_max_entry_count = std::min(current_max_entry_count, max_max_entry_count);
			const int current_max_entry_count_const = static_cast<int>(current_max_entry_count);

			std::map<std::string, plain_buffer::ptr> dedicated_buffers;
			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				dedicated_buffers.insert(std::make_pair(it->first, plain_buffer::ptr(new plain_buffer(it->second * current_max_entry_count))));

			plain_buffer::ptr temporary_working_fixed_buffer;
			if (temporary_working_fixed_size > 0)
				temporary_working_fixed_buffer = plain_buffer::ptr(new plain_buffer(temporary_working_fixed_size));

			std::vector<plain_buffer::ptr> layer_buffers;
			for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
				layer_buffers.push_back(plain_buffer::ptr(new plain_buffer(*it * current_max_entry_count)));

			unsigned int entry_processed_count = 0;

			while(true)
			{
				int entry_read_count = 0;
				#pragma omp parallel default(shared) num_threads(plain_config->openmp_thread_count) reduction(+:entry_read_count)
				{
					#pragma omp for schedule(dynamic)
					for(int entry_id = 0; entry_id < current_max_entry_count_const; ++entry_id)
					{
						std::map<std::string, float *> data_map;
						for(std::set<std::string>::const_iterator it = data_layer_names.begin(); it != data_layer_names.end(); ++it)
							data_map.insert(std::make_pair(*it, ((float *)(*dedicated_buffers[*it])) + entry_id * (dedicated_per_entry_data_name_to_size_map[*it] / sizeof(float))));
						if (reader.read(entry_processed_count + entry_id, data_map))
							++entry_read_count;
					}
				}
				if (entry_read_count == 0)
					break;

				for(std::vector<layer_name_with_action>::const_iterator action_it = actions_in_execution_order.begin(); action_it  != actions_in_execution_order.end(); ++action_it)
				{
					const layer_name_with_action& current_layer_name_with_action = *action_it;
					std::string layer_name = current_layer_name_with_action.get_name();;
					layer_action action = current_layer_name_with_action.get_action();
					layer::const_ptr current_layer = schema->find_layer(layer_name);

					plain_buffer::ptr output_buffer;
					{
						std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map.find(current_layer_name_with_action);
						if (it != layer_buffer_action_to_set_map.end())
							output_buffer = layer_buffers[it->second];
						else
							output_buffer = dedicated_buffers.find(layer_name)->second;
					}

					std::vector<plain_buffer::const_ptr> input_buffers;
					for(std::vector<std::string>::const_iterator input_layer_name_it = current_layer->input_layer_instance_names.begin(); input_layer_name_it != current_layer->input_layer_instance_names.end(); ++input_layer_name_it)
					{
						std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map.find(layer_name_with_action(*input_layer_name_it, layer_action::forward));
						if (it != layer_buffer_action_to_set_map.end())
							input_buffers.push_back(layer_buffers[it->second]);
						else
							input_buffers.push_back(dedicated_buffers.find(*input_layer_name_it)->second);
					}

					plain_buffer::ptr temporary_working_per_entry_buffer;
					{
						std::map<layer_name_with_action, unsigned int>::const_iterator it = temporary_working_per_entry_data_action_to_set_map.find(current_layer_name_with_action);
						if (it != temporary_working_per_entry_data_action_to_set_map.end())
							temporary_working_per_entry_buffer = layer_buffers[it->second];
					}

					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = current_layer->input_layer_instance_names.begin(); it2 != current_layer->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);

					testers.find(layer_name)->second->run_forward_propagation(
						output_buffer,
						input_buffers,
						temporary_working_fixed_buffer,
						temporary_working_per_entry_buffer,
						plain_config,
						current_layer,
						net_data->data_list.find(layer_name),
						net_data->data_custom_list.find(layer_name),
						input_layer_configuration_specific_list,
						layer_config_map[layer_name],
						entry_read_count * cumulative_tiling_factor_map[layer_name]);
				}

				for(int entry_id = 0; entry_id < entry_read_count; ++entry_id)
				{
					std::map<std::string, const float *> data_map;
					for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
						data_map.insert(std::make_pair(*it, ((float *)(*dedicated_buffers[*it])) + entry_id * (dedicated_per_entry_data_name_to_size_map[*it] / sizeof(float))));
					writer.write(data_map);
				}

				entry_processed_count += entry_read_count;

				if (entry_read_count < current_max_entry_count_const)
					break;
			}

			return entry_processed_count;
		}

		void forward_propagation_plain::layer_config_map_modified()
		{
			setup_dedicated_buffer_sizes();

			setup_layer_buffer_sizes();

			setup_temporary_working_fixed_buffer_sizes();

			update_max_entry_count();
		}

		void forward_propagation_plain::setup_dedicated_buffer_sizes()
		{
			dedicated_per_entry_data_name_to_size_map.clear();

			std::set<std::string> separate_buffers_layer_names(output_layer_names.begin(), output_layer_names.end());
			separate_buffers_layer_names.insert(data_layer_names.begin(), data_layer_names.end());
			for(std::set<std::string>::const_iterator it = separate_buffers_layer_names.begin(); it != separate_buffers_layer_names.end(); ++it)
				dedicated_per_entry_data_name_to_size_map.insert(std::make_pair(*it, layer_config_map.find(*it)->second.get_neuron_count() * cumulative_tiling_factor_map[*it] * sizeof(float)));
		}

		void forward_propagation_plain::setup_temporary_working_fixed_buffer_sizes()
		{
			temporary_working_fixed_size = 0;
			for(std::map<std::string, layer_tester_plain::const_ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
			{
				layer_configuration_specific output_layer_configuration_specific = layer_config_map[it->first];
				layer::const_ptr l = schema->get_layer(it->first);
				std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
					input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
				size_t new_temporary_working_fixed_size = it->second->get_temporary_working_fixed_buffer_size(
					plain_config,
					schema->get_layer(it->first),
					input_layer_configuration_specific_list,
					output_layer_configuration_specific);
				temporary_working_fixed_size = std::max(temporary_working_fixed_size, new_temporary_working_fixed_size);
			}

			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "forward prop plain working fixed buffer: " << ((temporary_working_fixed_size + 1024 - 1) / 1024) << " KB";
				debug->output_message(debug_str.str().c_str());
			}
		}

		void forward_propagation_plain::setup_layer_buffer_sizes()
		{
			std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > layer_buffer_set_list;
			{
				std::map<layer_name_with_action, unsigned int> input_index_layer_can_write_output_map;
				for(std::map<std::string, layer_tester_plain::const_ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
				{
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[it->first];
					layer::const_ptr l = schema->get_layer(it->first);
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
					int input_index_layer_can_write = it->second->get_input_index_layer_can_write(
						plain_config,
						schema->get_layer(it->first),
						input_layer_configuration_specific_list,
						output_layer_configuration_specific);
					if (input_index_layer_can_write >= 0)
						input_index_layer_can_write_output_map.insert(std::make_pair(layer_name_with_action(it->first, layer_action::forward), static_cast<unsigned int>(input_index_layer_can_write)));
				}

				std::map<layer_name_with_action, std::vector<buffer_lifetime> > buffers;
				std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<buffer_lifetime> > > dependencies;
				std::set<std::string> dedicated_output_buffers(output_layer_names.begin(), output_layer_names.end());
				for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
				{
					if (dedicated_output_buffers.find(it->get_name()) == dedicated_output_buffers.end())
						buffers.insert(std::make_pair(*it, std::vector<buffer_lifetime>(1, buffer_lifetime(buffer_lifetime::action_output_buffer))));
					layer::const_ptr l = schema->get_layer(it->get_name());
					std::map<layer_name_with_action, std::vector<buffer_lifetime> > current_dependencies;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
					{
						const std::string& previous_layer_name = *it2;
						if (data_layer_names.find(previous_layer_name) == data_layer_names.end())
							current_dependencies.insert(std::make_pair(layer_name_with_action(previous_layer_name, layer_action(layer_action::forward)), std::vector<buffer_lifetime>(1, buffer_lifetime(buffer_lifetime::action_output_buffer))));
					}
					if (!current_dependencies.empty())
						dependencies.insert(std::make_pair(*it, current_dependencies));
				}

				for(std::map<std::string, layer_tester_plain::const_ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
				{
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[it->first];
					layer::const_ptr l = schema->get_layer(it->first);
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
					size_t temporary_working_per_entry_buffer_size = it->second->get_temporary_working_per_entry_buffer_size(
						plain_config,
						schema->get_layer(it->first),
						input_layer_configuration_specific_list,
						output_layer_configuration_specific);
					if (temporary_working_per_entry_buffer_size > 0)
						buffers.insert(std::make_pair(layer_name_with_action(it->first, layer_action::forward), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::working_buffer));
				}

				layer_buffer_set_list = action_schema->get_buffer_set(
					buffers,
					dependencies,
					input_index_layer_can_write_output_map,
					std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >());
			}

			layer_buffer_set_per_entry_size_list.clear();
			layer_buffer_action_to_set_map.clear();
			temporary_working_per_entry_data_action_to_set_map.clear();
			for(unsigned int set_id = 0; set_id < layer_buffer_set_list.size(); ++set_id)
			{
				const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list[set_id];
				size_t max_buffer_size_per_entry = 0;
				for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
				{
					std::string layer_name = it->first.get_name();
					size_t buffer_size_per_entry;
					if (it->second.get_buffer_lifetime_type() == buffer_lifetime::action_output_buffer)
					{
						layer_buffer_action_to_set_map.insert(std::make_pair(it->first, set_id));
						buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
					}
					else if (it->second.get_buffer_lifetime_type() == buffer_lifetime::working_buffer)
					{
						temporary_working_per_entry_data_action_to_set_map.insert(std::make_pair(it->first, set_id));

						layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
						layer::const_ptr l = schema->get_layer(layer_name);
						std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
						for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
							input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
						size_t temporary_working_per_entry_buffer_size = testers.find(layer_name)->second->get_temporary_working_per_entry_buffer_size(
							plain_config,
							schema->get_layer(layer_name),
							input_layer_configuration_specific_list,
							output_layer_configuration_specific);

						buffer_size_per_entry = temporary_working_per_entry_buffer_size * cumulative_tiling_factor_map[layer_name];
					}
					else
						throw neural_network_exception((boost::format("Unexpected buffer lifetime %1% encountered for layer %2% action %3%") % it->second.str() % it->first.get_name() % it->first.get_action().str()).str());
					max_buffer_size_per_entry = std::max(max_buffer_size_per_entry, buffer_size_per_entry);
				}
				layer_buffer_set_per_entry_size_list.push_back(max_buffer_size_per_entry);
			}
			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "forward prop plain per entry buffers: " << layer_buffer_set_per_entry_size_list.size();
				if (!layer_buffer_set_per_entry_size_list.empty())
				{
					debug_str << " (";
					for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
					{
						if (it != layer_buffer_set_per_entry_size_list.begin())
							debug_str << ", ";
						debug_str << ((*it + 1024 - 1) / 1024) << " KB";
					}
					debug_str << ")";
				}
				debug->output_message(debug_str.str().c_str());
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_plain_per_entry_buffers", "gv"), std::ios_base::out | std::ios_base::trunc);
				action_schema->write_gv(out, layer_buffer_action_to_set_map, std::map<layer_name_with_action, unsigned int>(), temporary_working_per_entry_data_action_to_set_map);
			}
		}

		void forward_propagation_plain::update_max_entry_count()
		{
			buffer_plain_size_configuration buffer_configuration;

			std::vector<std::string> data_name_list = net_data->data_list.get_data_layer_name_list();
			for(std::vector<std::string>::const_iterator it = data_name_list.begin(); it != data_name_list.end(); ++it)
			{
				layer_data::ptr d = net_data->data_list.get(*it);
				for(layer_data::const_iterator it2 = d->begin(); it2 != d->end(); ++it2)
					buffer_configuration.add_constant_buffer(it2->size() * sizeof(float));
			}

			std::vector<std::string> data_custom_name_list = net_data->data_custom_list.get_data_custom_layer_name_list();
			for(std::vector<std::string>::const_iterator it = data_custom_name_list.begin(); it != data_custom_name_list.end(); ++it)
			{
				layer_data_custom::ptr d = net_data->data_custom_list.get(*it);
				for(layer_data_custom::const_iterator it2 = d->begin(); it2 != d->end(); ++it2)
					buffer_configuration.add_constant_buffer(it2->size() * sizeof(int));
			}

			for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				buffer_configuration.add_per_entry_buffer(it->second);

			buffer_configuration.add_constant_buffer(temporary_working_fixed_size);

			max_entry_count = plain_config->get_max_entry_count(buffer_configuration);

			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "forward prop plain max packet size: " << max_entry_count;
				if (max_entry_count > max_max_entry_count)
					debug_str << ", will be capped by " << max_max_entry_count;
				debug->output_message(debug_str.str().c_str());
			}
		}
	}
}
