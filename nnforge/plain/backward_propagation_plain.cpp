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

#include "backward_propagation_plain.h"

#include "layer_updater_plain_factory.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/fstream.hpp>

#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		backward_propagation_plain::backward_propagation_plain(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			const std::vector<std::string>& error_source_layer_names,
			const std::vector<std::string>& exclude_data_update_layer_names,
			debug_state::ptr debug,
			plain_running_configuration::const_ptr plain_config)
			: backward_propagation(schema, output_layer_names, error_source_layer_names, exclude_data_update_layer_names, debug)
			, plain_config(plain_config)
			, temporary_working_fixed_size(0)
		{
			actions_in_execution_order = action_schema->get_actions_in_execution_order();

			for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
			{
				if (it->get_action().get_action_type() != layer_action::backward_data)
					continue;
				layer::const_ptr l = this->schema->get_layer(it->get_name());
				const std::string& previous_layer_name = l->input_layer_instance_names[it->get_action().get_backprop_index()];
				input_to_random_output_map.insert(std::make_pair(previous_layer_name, *it));
			}

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
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("backward_prop_plain_action_schema_sequential", "gv"), std::ios_base::out | std::ios_base::trunc);
				action_schema->write_gv(out);
			}

			std::set<std::string> action_layer_names;
			for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
			{
				action_layer_names.insert(it->get_name());
				layer_name_to_action_set_map.insert(std::make_pair(it->get_name(), std::set<layer_action>())).first->second.insert(it->get_action());
			}
			for(std::set<std::string> ::const_iterator it = action_layer_names.begin(); it != action_layer_names.end(); ++it)
				updaters.insert(
					std::make_pair(
						*it,
						layer_updater_plain_factory::singleton::get_const_instance().get_updater_plain_layer(this->schema->get_layer(*it)->get_type_name())));
		}

		backward_propagation_plain::~backward_propagation_plain()
		{
		}

		std::pair<unsigned int, std::map<std::string, std::vector<float> > > backward_propagation_plain::actual_run(
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
			std::map<std::string, std::vector<double> > updates_accumulated;
			std::vector<std::string> data_layer_list = data.data_list.get_data_layer_name_list();
			for(std::vector<std::string>::const_iterator it = data_layer_list.begin(); it != data_layer_list.end(); ++it)
			{
				const std::string& layer_name = *it;
				layer_data::ptr d = data.data_list.get(layer_name);
				updates_accumulated.insert(std::make_pair(layer_name, std::vector<double>(d->size(), 0.0)));
			}

			std::vector<layer::const_ptr> layer_list;
			for(std::vector<std::string>::const_iterator it = data_layer_list.begin(); it != data_layer_list.end(); ++it)
				layer_list.push_back(schema->get_layer(*it));
			layer_data_list::ptr gradient(new layer_data_list(layer_list, 0.0F));

			buffer_plain_size_configuration buffer_configuration = buffer_config_without_data_and_momentum;
			{
				for(std::vector<std::string>::const_iterator it = data_layer_list.begin(); it != data_layer_list.end(); ++it)
				{
					const std::string& layer_name = *it;
					layer_data::ptr d = data.data_list.get(layer_name);
					for(layer_data::const_iterator it2 = d->begin(); it2 != d->end(); ++it2)
					{
						buffer_configuration.add_constant_buffer(it2->size() * sizeof(float)); // data
						buffer_configuration.add_constant_buffer(it2->size() * sizeof(float)); // gradient
						if (momentum.is_momentum_data())
							buffer_configuration.add_constant_buffer(it2->size() * sizeof(float)); // momentum
						if (momentum.is_momentum_data2())
							buffer_configuration.add_constant_buffer(it2->size() * sizeof(float)); // 2nd momentum
					}
				}
				std::vector<std::string> data_custom_layer_list = data.data_custom_list.get_data_custom_layer_name_list();
				for(std::vector<std::string>::const_iterator it = data_custom_layer_list.begin(); it != data_custom_layer_list.end(); ++it)
				{
					const std::string& layer_name = *it;
					layer_data_custom::ptr d = data.data_custom_list.get(layer_name);
					for(layer_data_custom::const_iterator it2 = d->begin(); it2 != d->end(); ++it2)
						buffer_configuration.add_constant_buffer(it2->size() * sizeof(int));
				}
				for(std::map<std::string, std::vector<double> >::const_iterator it = updates_accumulated.begin(); it != updates_accumulated.end(); ++it)
					buffer_configuration.add_constant_buffer(it->second.size() * sizeof(double));
			}

			unsigned int max_entry_count = plain_config->get_max_entry_count(buffer_configuration);

			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "backward prop plain max packet size: " << max_entry_count;
				debug->output_message(debug_str.str().c_str());
			}

			if (max_entry_count == 0)
				throw neural_network_exception("Insufficient memory to do forward-backward prop for even one sample");

			std::vector<unsigned int> entry_read_count_list;
			if (batch_size <= max_entry_count)
				entry_read_count_list.push_back(batch_size);
			else
			{
				unsigned int chunk_count = (batch_size + max_entry_count - 1) / max_entry_count;
				unsigned int chunk_min_size = batch_size / chunk_count;
				unsigned int plus1_chunk_count = batch_size % chunk_count;
				entry_read_count_list.resize(chunk_count);
				std::fill_n(entry_read_count_list.begin(), plus1_chunk_count, chunk_min_size + 1);
				std::fill_n(entry_read_count_list.begin() + plus1_chunk_count, chunk_count - plus1_chunk_count, chunk_min_size);

				if (debug->is_debug())
				{
					std::stringstream debug_str;
					debug_str << "Batch " << batch_size << " is split into multiple chunks: ";
					for(std::vector<unsigned int>::const_iterator it = entry_read_count_list.begin(); it != entry_read_count_list.end(); ++it)
					{
						if (it != entry_read_count_list.begin())
							debug_str << ", ";
						debug_str << *it;
					}
					debug->output_message(debug_str.str().c_str());
				}
			}
			unsigned int max_chunk_size = *std::max_element(entry_read_count_list.begin(), entry_read_count_list.end());

			std::map<std::string, plain_buffer::ptr> dedicated_buffers;
			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				dedicated_buffers.insert(std::make_pair(it->first, plain_buffer::ptr(new plain_buffer(it->second * max_chunk_size))));

			plain_buffer::ptr temporary_working_fixed_buffer;
			if (temporary_working_fixed_size > 0)
				temporary_working_fixed_buffer = plain_buffer::ptr(new plain_buffer(temporary_working_fixed_size));

			std::vector<plain_buffer::ptr> layer_buffers;
			for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
				layer_buffers.push_back(plain_buffer::ptr(new plain_buffer(*it * max_chunk_size)));

			unsigned int base_iteration_count = 0;
			if (momentum.type == training_momentum::adam_momentum)
			{
				int epoch_entry_count = reader.get_entry_count();
				if (epoch_entry_count >= 0)
					base_iteration_count = epoch_id * ((epoch_entry_count + batch_size - 1) / batch_size);
				else
					throw neural_network_exception("Training data reader doesn't report entry_count, which is required for ADAM momentum");
			}

			unsigned int entry_processed_count = 0;
			unsigned int chunk_index = 0;
			unsigned int gradient_accumulated_entry_count = 0;
			unsigned int gradient_applied_count = 0;

			while(true)
			{
				const int current_max_entry_count_const = entry_read_count_list[chunk_index];
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

				gradient_accumulated_entry_count += entry_read_count;
				bool is_apply_gradient = false;
				float gradient_normalizer;
				if (gradient_accumulated_entry_count >= batch_size)
				{
					is_apply_gradient = true;
					gradient_normalizer = 1.0F / static_cast<float>(gradient_accumulated_entry_count);
					gradient_accumulated_entry_count = 0;
					gradient_applied_count++;
				}

				for(std::vector<layer_name_with_action>::const_iterator action_it = actions_in_execution_order.begin(); action_it  != actions_in_execution_order.end(); ++action_it)
				{
					const layer_name_with_action& current_layer_name_with_action = *action_it;
					std::string layer_name = current_layer_name_with_action.get_name();;
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
					layer::const_ptr l = schema->get_layer(layer_name);
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
					layer_action action = current_layer_name_with_action.get_action();
					layer::const_ptr current_layer = schema->find_layer(layer_name);
					const std::set<layer_action>& actions = layer_name_to_action_set_map[layer_name];
					unsigned int tiling_factor = cumulative_tiling_factor_map[layer_name];

					plain_buffer::ptr temporary_working_per_entry_buffer;
					{
						std::map<layer_name_with_action, unsigned int>::const_iterator it = temporary_working_per_entry_data_action_to_set_map.find(current_layer_name_with_action);
						if (it != temporary_working_per_entry_data_action_to_set_map.end())
							temporary_working_per_entry_buffer = layer_buffers[it->second];
					}

					switch (action.get_action_type())
					{
					case layer_action::forward:
						{
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

							plain_buffer::ptr temporary_per_entry_buffer;
							{
								std::map<layer_name_with_action, unsigned int>::const_iterator it = temporary_per_entry_data_action_to_set_map.find(current_layer_name_with_action);
								if (it != temporary_per_entry_data_action_to_set_map.end())
									temporary_per_entry_buffer = layer_buffers[it->second];
							}

							updaters.find(layer_name)->second->run_forward_propagation(
								output_buffer,
								input_buffers,
								temporary_working_fixed_buffer,
								temporary_working_per_entry_buffer,
								temporary_per_entry_buffer,
								plain_config,
								current_layer,
								data.data_list.find(layer_name),
								data.data_custom_list.find(layer_name),
								input_layer_configuration_specific_list,
								output_layer_configuration_specific,
								actions,
								entry_read_count * tiling_factor);
						}
						break;
					case layer_action::backward_data:
						{
							plain_buffer::ptr output_buffer = layer_buffers[layer_buffer_action_to_set_map[current_layer_name_with_action]];

							std::vector<plain_buffer::const_ptr> input_neurons_buffers;
							unsigned int data_input_index = 0;
							for(std::vector<std::string>::const_iterator input_layer_name_it = current_layer->input_layer_instance_names.begin(); input_layer_name_it != current_layer->input_layer_instance_names.end(); ++input_layer_name_it, ++data_input_index)
							{
								if (updaters[layer_name]->is_backward_data_dependent_on_input_buffer(action.get_backprop_index(), data_input_index, actions, plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
								{
									std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map.find(layer_name_with_action(*input_layer_name_it, layer_action::forward));
									if (it != layer_buffer_action_to_set_map.end())
										input_neurons_buffers.push_back(layer_buffers[it->second]);
									else
										input_neurons_buffers.push_back(dedicated_buffers[*input_layer_name_it]);
								}
								else
									input_neurons_buffers.push_back(plain_buffer::const_ptr());
							}

							plain_buffer::ptr temporary_per_entry_buffer;
							{
								if (updaters[layer_name]->is_backward_data_dependent_on_temporary_per_entry_buffer(action.get_backprop_index(), actions, plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
								{
									std::map<layer_name_with_action, unsigned int>::const_iterator it = temporary_per_entry_data_action_to_set_map.find(layer_name_with_action(layer_name, layer_action::forward));
									if (it != temporary_per_entry_data_action_to_set_map.end())
										temporary_per_entry_buffer = layer_buffers[it->second];
								}
							}

							plain_buffer::const_ptr output_neurons_buffer;
							{
								if (updaters[layer_name]->is_backward_data_dependent_on_output_buffer(action.get_backprop_index(), actions, plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
								{
									std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map.find(layer_name_with_action(layer_name, layer_action::forward));
									if (it != layer_buffer_action_to_set_map.end())
										output_neurons_buffer = layer_buffers[it->second];
									else
										output_neurons_buffer = dedicated_buffers[layer_name];
								}
							}

							plain_buffer::const_ptr output_errors_buffer;
							{
								std::map<std::string, layer_name_with_action>::const_iterator it = input_to_random_output_map.find(layer_name);
								if (it != input_to_random_output_map.end())
									output_errors_buffer = layer_buffers[layer_buffer_action_to_set_map[it->second]];
							}

							updaters.find(layer_name)->second->run_backward_data_propagation(
								action.get_backprop_index(),
								output_buffer,
								output_errors_buffer,
								input_neurons_buffers,
								output_neurons_buffer,
								temporary_working_fixed_buffer,
								temporary_working_per_entry_buffer,
								temporary_per_entry_buffer,
								plain_config,
								current_layer,
								data.data_list.find(layer_name),
								data.data_custom_list.find(layer_name),
								input_layer_configuration_specific_list,
								output_layer_configuration_specific,
								add_output_actions.find(current_layer_name_with_action) != add_output_actions.end(),
								actions,
								entry_read_count * tiling_factor);
						}
						break;
					case layer_action::backward_weights:
						{
							std::vector<plain_buffer::const_ptr> input_neurons_buffers;
							unsigned int data_input_index = 0;
							for(std::vector<std::string>::const_iterator input_layer_name_it = current_layer->input_layer_instance_names.begin(); input_layer_name_it != current_layer->input_layer_instance_names.end(); ++input_layer_name_it, ++data_input_index)
							{
								if (updaters[layer_name]->is_backward_weights_dependent_on_input_buffer(data_input_index, actions, plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
								{
									std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map.find(layer_name_with_action(*input_layer_name_it, layer_action::forward));
									if (it != layer_buffer_action_to_set_map.end())
										input_neurons_buffers.push_back(layer_buffers[it->second]);
									else
										input_neurons_buffers.push_back(dedicated_buffers[*input_layer_name_it]);
								}
								else
									input_neurons_buffers.push_back(plain_buffer::const_ptr());
							}

							plain_buffer::ptr temporary_per_entry_buffer;
							{
								if (updaters[layer_name]->is_backward_weights_dependent_on_temporary_per_entry_buffer(actions, plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
								{
									std::map<layer_name_with_action, unsigned int>::const_iterator it = temporary_per_entry_data_action_to_set_map.find(layer_name_with_action(layer_name, layer_action::forward));
									if (it != temporary_per_entry_data_action_to_set_map.end())
										temporary_per_entry_buffer = layer_buffers[it->second];
								}
							}

							plain_buffer::const_ptr output_errors_buffer;
							{
								std::map<std::string, layer_name_with_action>::const_iterator it = input_to_random_output_map.find(layer_name);
								if (it != input_to_random_output_map.end())
									output_errors_buffer = layer_buffers[layer_buffer_action_to_set_map[it->second]];
							}

							updaters.find(layer_name)->second->run_backward_weights_propagation(
								input_neurons_buffers,
								output_errors_buffer,
								temporary_working_fixed_buffer,
								temporary_working_per_entry_buffer,
								temporary_per_entry_buffer,
								plain_config,
								current_layer,
								gradient->find(layer_name),
								data.data_custom_list.find(layer_name),
								input_layer_configuration_specific_list,
								output_layer_configuration_specific,
								actions,
								entry_read_count * tiling_factor);
						}
						break;
					case layer_action::update_weights:
						{
							if (is_apply_gradient)
							{
								layer_data::ptr previous_upd;
								if (momentum.is_momentum_data())
									previous_upd = momentum_data->data_list.find(layer_name);
								layer_data::ptr previous_upd2;
								if (momentum.is_momentum_data2())
									previous_upd2 = momentum_data2->data_list.find(layer_name);
								apply_gradient(
									layer_name,
									data.data_list.find(layer_name),
									gradient->find(layer_name),
									previous_upd,
									previous_upd2,
									updates_accumulated[layer_name],
									learning_rates.find(layer_name)->second,
									gradient_normalizer,
									weight_decay,
									momentum,
									base_iteration_count + gradient_applied_count);
							}
						}
						break;
					}
				}

				for(int entry_id = 0; entry_id < entry_read_count * static_cast<int>(output_layers_tiling_factor); ++entry_id)
				{
					std::map<std::string, const float *> data_map;
					for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
						data_map.insert(std::make_pair(*it, ((float *)(*dedicated_buffers[*it])) + entry_id * (dedicated_per_entry_data_name_to_size_map[*it] / sizeof(float) / output_layers_tiling_factor)));
					writer.write(data_map);
				}

				entry_processed_count += entry_read_count;
				chunk_index = (chunk_index + 1) % entry_read_count_list.size();

				if (entry_read_count < current_max_entry_count_const)
					break;
			}

			if (gradient_accumulated_entry_count > 0)
			{
				float gradient_normalizer = 1.0F / static_cast<float>(batch_size);
				gradient_applied_count++;
				for(std::map<std::string, std::vector<double> >::const_iterator it = updates_accumulated.begin(); it != updates_accumulated.end(); ++it)
				{
					const std::string& layer_name = it->first;
					layer_data::ptr previous_upd;
					if (momentum.is_momentum_data())
						previous_upd = momentum_data->data_list.find(layer_name);
					layer_data::ptr previous_upd2;
					if (momentum.is_momentum_data2())
						previous_upd2 = momentum_data2->data_list.find(layer_name);
					apply_gradient(
						layer_name,
						data.data_list.find(layer_name),
						gradient->find(layer_name),
						previous_upd,
						previous_upd2,
						updates_accumulated[layer_name],
						learning_rates.find(layer_name)->second,
						gradient_normalizer,
						weight_decay,
						momentum,
						base_iteration_count + gradient_applied_count);
				}
			}

			std::map<std::string, std::vector<float> > updates_accumulated_f;
			{
				float mult = 1.0F / static_cast<float>(gradient_applied_count);
				for(std::map<std::string, std::vector<double> >::const_iterator it = updates_accumulated.begin(); it != updates_accumulated.end(); ++it)
				{
					std::vector<float>& f = updates_accumulated_f.insert(std::make_pair(it->first, std::vector<float>())).first->second;
					layer_data::const_iterator it_data = data.data_list.find(it->first)->begin();
					for(std::vector<double>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2, ++it_data)
						f.push_back(static_cast<float>(*it2) * mult / static_cast<float>(it_data->size()));
				}
			}

			return std::make_pair(entry_processed_count, updates_accumulated_f);
		}

		void backward_propagation_plain::layer_config_map_modified()
		{
			setup_dedicated_buffer_sizes();

			setup_layer_buffer_sizes();

			setup_temporary_working_fixed_buffer_sizes();

			update_buffer_config();
		}

		void backward_propagation_plain::setup_dedicated_buffer_sizes()
		{
			dedicated_per_entry_data_name_to_size_map.clear();

			std::set<std::string> separate_buffers_layer_names(output_layer_names.begin(), output_layer_names.end());
			separate_buffers_layer_names.insert(data_layer_names.begin(), data_layer_names.end());
			for(std::set<std::string>::const_iterator it = separate_buffers_layer_names.begin(); it != separate_buffers_layer_names.end(); ++it)
				dedicated_per_entry_data_name_to_size_map.insert(std::make_pair(*it, layer_config_map.find(*it)->second.get_neuron_count() * cumulative_tiling_factor_map[*it] * sizeof(float)));
		}

		void backward_propagation_plain::setup_temporary_working_fixed_buffer_sizes()
		{
			temporary_working_fixed_size = 0;
			for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
			{
				std::string layer_name = it->get_name();
				layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
				layer::const_ptr l = schema->get_layer(layer_name);
				std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
					input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
				size_t new_temporary_working_fixed_size = updaters[layer_name]->get_temporary_working_fixed_buffer_size(
					it->get_action(),
					layer_name_to_action_set_map[layer_name],
					plain_config,
					l,
					input_layer_configuration_specific_list,
					output_layer_configuration_specific);
				temporary_working_fixed_size = std::max(temporary_working_fixed_size, new_temporary_working_fixed_size);
			}

			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "backward prop plain working fixed buffer: " << ((temporary_working_fixed_size + 1024 - 1) / 1024) << " KB";
				debug->output_message(debug_str.str().c_str());
			}
		}

		void backward_propagation_plain::setup_layer_buffer_sizes()
		{
			std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > layer_buffer_set_list;
			{
				std::map<layer_name_with_action, unsigned int> input_index_layer_can_write_output_map;
				for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
				{
					std::string layer_name = it->get_name();
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
					layer::const_ptr l = schema->get_layer(layer_name);
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);
					int input_index_layer_can_write = updaters[it->get_name()]->get_input_index_layer_can_write(
						it->get_action(),
						layer_name_to_action_set_map[layer_name],
						plain_config,
						l,
						input_layer_configuration_specific_list,
						output_layer_configuration_specific);
					if (input_index_layer_can_write >= 0)
						input_index_layer_can_write_output_map.insert(std::make_pair(*it, static_cast<unsigned int>(input_index_layer_can_write)));
				}

				std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > > buffers;
				std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<buffer_lifetime> > > dependencies;
				std::set<std::string> dedicated_output_buffers(output_layer_names.begin(), output_layer_names.end());
				for(std::vector<layer_name_with_action>::const_iterator it = actions_in_execution_order.begin(); it != actions_in_execution_order.end(); ++it)
				{
					std::string layer_name = it->get_name();
					layer::const_ptr l = schema->get_layer(layer_name);
					layer_updater_plain::const_ptr updater = updaters[layer_name];
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);

					std::vector<std::pair<buffer_lifetime, float> > current_buffers;
					{
						switch (it->get_action().get_action_type())
						{
						case layer_action::forward:
							{
								size_t buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
								if (dedicated_output_buffers.find(it->get_name()) == dedicated_output_buffers.end())
										current_buffers.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::action_output_buffer), static_cast<float>(buffer_size_per_entry)));
							}
							{
								size_t temporary_per_entry_buffer_size = updater->get_temporary_per_entry_buffer_size(
									layer_name_to_action_set_map[layer_name],
									plain_config,
									l,
									input_layer_configuration_specific_list,
									output_layer_configuration_specific) * cumulative_tiling_factor_map[layer_name];
								if (temporary_per_entry_buffer_size > 0)
									current_buffers.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::temporary_buffer), static_cast<float>(temporary_per_entry_buffer_size)));
							}
							break;
						case layer_action::backward_data:
							{
								const std::string& previous_layer_name = schema->get_layer(layer_name)->input_layer_instance_names[it->get_action().get_backprop_index()];
								size_t buffer_size_per_entry = layer_config_map.find(previous_layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[previous_layer_name] * sizeof(float);
								current_buffers.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::action_output_buffer), static_cast<float>(buffer_size_per_entry)));
							}
							break;
						}

						{
							size_t temporary_working_per_entry_buffer_size = updater->get_temporary_working_per_entry_buffer_size(
								it->get_action(),
								layer_name_to_action_set_map[layer_name],
								plain_config,
								l,
								input_layer_configuration_specific_list,
								output_layer_configuration_specific) * cumulative_tiling_factor_map[layer_name];
							if (temporary_working_per_entry_buffer_size > 0)
								current_buffers.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::working_buffer), static_cast<float>(temporary_working_per_entry_buffer_size)));
						}
					}

					if (!current_buffers.empty())
						buffers.insert(std::make_pair(*it, current_buffers));

					std::map<layer_name_with_action, std::vector<buffer_lifetime> > current_dependencies;
					{
						layer::const_ptr l = schema->get_layer(it->get_name());
						switch (it->get_action().get_action_type())
						{
						case layer_action::forward:
							{
								for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
								{
									const std::string& previous_layer_name = *it2;
									if (data_layer_names.find(previous_layer_name) == data_layer_names.end())
										current_dependencies.insert(std::make_pair(layer_name_with_action(previous_layer_name, layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
								}
							}
							break;
						case layer_action::backward_weights:
							{
								unsigned int data_input_index = 0;
								for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2, ++data_input_index)
								{
									const std::string& previous_layer_name = *it2;
									if ((data_layer_names.find(previous_layer_name) == data_layer_names.end()) &&
										updater->is_backward_weights_dependent_on_input_buffer(data_input_index, layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
									{
										current_dependencies.insert(std::make_pair(layer_name_with_action(previous_layer_name, layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
									}
								}
								std::map<std::string, layer_name_with_action>::const_iterator input_to_random_output_it = input_to_random_output_map.find(l->instance_name);
								if (input_to_random_output_it != input_to_random_output_map.end())
									current_dependencies.insert(std::make_pair(input_to_random_output_it->second, std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
								if (updater->is_backward_weights_dependent_on_temporary_per_entry_buffer(layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
									current_dependencies.insert(std::make_pair(layer_name_with_action(it->get_name(), layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::temporary_buffer));
							}
							break;
						case layer_action::backward_data:
							{
								unsigned int action_input_index = it->get_action().get_backprop_index();
								unsigned int data_input_index = 0;
								for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2, ++data_input_index)
								{
									const std::string& previous_layer_name = *it2;
									if ((data_layer_names.find(previous_layer_name) == data_layer_names.end()) && updater->is_backward_data_dependent_on_input_buffer(action_input_index, data_input_index, layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
										current_dependencies.insert(std::make_pair(layer_name_with_action(previous_layer_name, layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
								}
								if (updater->is_backward_data_dependent_on_output_buffer(action_input_index, layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
									current_dependencies.insert(std::make_pair(layer_name_with_action(it->get_name(), layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
								std::map<std::string, layer_name_with_action>::const_iterator input_to_random_output_it = input_to_random_output_map.find(l->instance_name);
								if (input_to_random_output_it != input_to_random_output_map.end())
									current_dependencies.insert(std::make_pair(input_to_random_output_it->second, std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::action_output_buffer));
								if (updater->is_backward_data_dependent_on_temporary_per_entry_buffer(action_input_index, layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific))
									current_dependencies.insert(std::make_pair(layer_name_with_action(it->get_name(), layer_action(layer_action::forward)), std::vector<buffer_lifetime>())).first->second.push_back(buffer_lifetime(buffer_lifetime::temporary_buffer));
							}
							break;
						}
					}

					if (!current_dependencies.empty())
						dependencies.insert(std::make_pair(*it, current_dependencies));
				}

				std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > should_be_placed_into_the_same_buffers;
				for(std::vector<std::vector<layer_name_with_action> >::const_iterator it = same_output_action_sets.begin(); it != same_output_action_sets.end(); ++it)
				{
					const std::vector<layer_name_with_action>& src_tt = *it;
					should_be_placed_into_the_same_buffers.push_back(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >());
					std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& tt = should_be_placed_into_the_same_buffers.back();
					for(std::vector<layer_name_with_action>::const_iterator it2 = src_tt.begin(); it2 != src_tt.end(); ++it2)
						tt.push_back(std::make_pair(*it2, buffer_lifetime(buffer_lifetime::action_output_buffer)));
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
			temporary_per_entry_data_action_to_set_map.clear();
			for(unsigned int set_id = 0; set_id < layer_buffer_set_list.size(); ++set_id)
			{
				const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list[set_id];
				size_t max_buffer_size_per_entry = 0;
				for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
				{
					std::string layer_name = it->first.get_name();
					layer::const_ptr l = schema->get_layer(layer_name);
					layer_updater_plain::const_ptr updater = updaters[layer_name];
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);

					size_t buffer_size_per_entry;
					switch(it->second.get_buffer_lifetime_type())
					{
					case buffer_lifetime::action_output_buffer:
						layer_buffer_action_to_set_map.insert(std::make_pair(it->first, set_id));
						switch (it->first.get_action().get_action_type())
						{
						case layer_action::forward:
							buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
							break;
						case layer_action::backward_data:
							{
								const std::string& previous_layer_name = schema->get_layer(layer_name)->input_layer_instance_names[it->first.get_action().get_backprop_index()];
								buffer_size_per_entry = layer_config_map.find(previous_layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[previous_layer_name] * sizeof(float);
							}
							break;
						default:
							throw neural_network_exception((boost::format("Unexpected buffer lifetime %1% encountered for layer %2% action %3%") % it->second.str() % it->first.get_name() % it->first.get_action().str()).str());
						}
						break;
					case buffer_lifetime::working_buffer:
						temporary_working_per_entry_data_action_to_set_map.insert(std::make_pair(it->first, set_id));
						buffer_size_per_entry = updaters[layer_name]->get_temporary_working_per_entry_buffer_size(it->first.get_action(), layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific) * cumulative_tiling_factor_map[layer_name];
						break;
					case buffer_lifetime::temporary_buffer:
						temporary_per_entry_data_action_to_set_map.insert(std::make_pair(it->first, set_id));
						buffer_size_per_entry = updaters[layer_name]->get_temporary_per_entry_buffer_size(layer_name_to_action_set_map[layer_name], plain_config, l, input_layer_configuration_specific_list, output_layer_configuration_specific) * cumulative_tiling_factor_map[layer_name];
						break;
					default:
						throw neural_network_exception((boost::format("Unexpected buffer lifetime %1% encountered for layer %2% action %3%") % it->second.str() % it->first.get_name() % it->first.get_action().str()).str());
					}
					max_buffer_size_per_entry = std::max(max_buffer_size_per_entry, buffer_size_per_entry);
				}
				layer_buffer_set_per_entry_size_list.push_back(max_buffer_size_per_entry);
			}
			if (debug->is_debug())
			{
				std::stringstream debug_str;
				debug_str << "backward prop plain per entry buffers: " << layer_buffer_set_per_entry_size_list.size();
				size_t total_buffer_size = 0;
				for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
						total_buffer_size += *it;
				debug_str << ", total size " << ((total_buffer_size + 1024 - 1) / 1024) << " KB";
				debug->output_message(debug_str.str().c_str());
				for(unsigned int set_id = 0; set_id < static_cast<unsigned int>(layer_buffer_set_per_entry_size_list.size()); ++set_id)
				{
					std::stringstream debug_str;
					debug_str << " - " << ((layer_buffer_set_per_entry_size_list[set_id] + 1024 - 1) / 1024) << " KB: ";
					const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list[set_id];
					for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
					{
						if (it != action_list.begin())
							debug_str << ", ";
						debug_str << it->first.get_name() << " " << it->first.get_action().str();
						if (it->second.get_buffer_lifetime_type() != buffer_lifetime::action_output_buffer)
							debug_str << " " << it->second.str();
					}
					debug->output_message(debug_str.str().c_str());
				}
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("backward_prop_plain_per_entry_buffers", "gv"), std::ios_base::out | std::ios_base::trunc);
				action_schema->write_gv(out, layer_buffer_action_to_set_map, temporary_per_entry_data_action_to_set_map, temporary_working_per_entry_data_action_to_set_map);
			}
		}

		void backward_propagation_plain::update_buffer_config()
		{
			buffer_plain_size_configuration buffer_configuration;

			for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				buffer_configuration.add_per_entry_buffer(it->second);

			buffer_configuration.add_constant_buffer(temporary_working_fixed_size);

			buffer_config_without_data_and_momentum = buffer_configuration;
		}

		void backward_propagation_plain::apply_gradient(
			const std::string& layer_name,
			layer_data::ptr data,
			layer_data::ptr gradient,
			layer_data::ptr previous_upd,
			layer_data::ptr previous_upd2,
			std::vector<double>& updates_accumulated,
			const std::vector<float>& learning_rates,
			float normalizer,
			float weight_decay,
			training_momentum momentum,
			unsigned int iteration_id) const
		{
			switch (momentum.type)
			{
			case training_momentum::no_momentum:
				{
					layer_data::iterator gradient_it = gradient->begin();
					std::vector<double>::iterator updates_accumulated_it = updates_accumulated.begin();
					std::vector<float>::const_iterator learning_rate_it = learning_rates.begin();
					std::set<unsigned int> weight_decay_part_id_set = schema->get_layer(layer_name)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					double accum = 0.0;
					for(layer_data::iterator data_it = data->begin(); data_it != data->end(); ++data_it, ++gradient_it, ++learning_rate_it, ++part_id, ++updates_accumulated_it)
					{
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						float learning_rate = *learning_rate_it;
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2)
						{
							float current_weight = *data_it2;
							float gr = *gradient_it2;
							float upd = learning_rate * (gr * normalizer - current_weight * actual_weight_decay);
							accum += static_cast<double>(fabsf(upd));
							float new_weight = current_weight + upd;
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
						}
						*updates_accumulated_it += accum;
					}
				}
				break;
			case training_momentum::vanilla_momentum:
				{
					layer_data::iterator gradient_it = gradient->begin();
					layer_data::iterator previous_upd_it = previous_upd->begin();
					std::vector<double>::iterator updates_accumulated_it = updates_accumulated.begin();
					std::vector<float>::const_iterator learning_rate_it = learning_rates.begin();
					std::set<unsigned int> weight_decay_part_id_set = schema->get_layer(layer_name)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					for(layer_data::iterator data_it = data->begin(); data_it != data->end(); ++data_it, ++gradient_it, ++previous_upd_it, ++learning_rate_it, ++part_id, ++updates_accumulated_it)
					{
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						std::vector<float>::iterator previous_upd_it2 = previous_upd_it->begin();
						float learning_rate = *learning_rate_it;
						double accum = 0.0;
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++previous_upd_it2)
						{
							float current_weight = *data_it2;
							float gr = *gradient_it2;
							float prev_upd = *previous_upd_it2;
							float upd = prev_upd * momentum.momentum_val + learning_rate * (gr * normalizer - current_weight * actual_weight_decay);
							accum += static_cast<double>(fabsf(upd));
							float new_weight = current_weight + upd;
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
							*previous_upd_it2 = upd;
						}
						*updates_accumulated_it += accum;
					}
				}
				break;
			case training_momentum::nesterov_momentum:
				{
					layer_data::iterator gradient_it = gradient->begin();
					layer_data::iterator previous_upd_it = previous_upd->begin();
					std::vector<double>::iterator updates_accumulated_it = updates_accumulated.begin();
					std::vector<float>::const_iterator learning_rate_it = learning_rates.begin();
					std::set<unsigned int> weight_decay_part_id_set = schema->get_layer(layer_name)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					for(layer_data::iterator data_it = data->begin(); data_it != data->end(); ++data_it, ++gradient_it, ++previous_upd_it, ++learning_rate_it, ++part_id, ++updates_accumulated_it)
					{
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						std::vector<float>::iterator previous_upd_it2 = previous_upd_it->begin();
						float learning_rate = *learning_rate_it;
						double accum = 0.0;
						float mp1 = momentum.momentum_val + 1.0F;
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++previous_upd_it2)
						{
							float current_weight = *data_it2;
							float gr = *gradient_it2;
							float prev_upd = *previous_upd_it2;
							float new_upd = prev_upd * momentum.momentum_val + learning_rate * (gr * normalizer - current_weight * actual_weight_decay);
							float upd = mp1 * new_upd - momentum.momentum_val * prev_upd;
							accum += static_cast<double>(fabsf(upd));
							float new_weight = current_weight + upd;
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
							*previous_upd_it2 = new_upd;
						}
						*updates_accumulated_it += accum;
					}
				}
				break;
			case training_momentum::adam_momentum:
				{
					layer_data::iterator gradient_it = gradient->begin();
					layer_data::iterator previous_upd_it = previous_upd->begin();
					layer_data::iterator previous_upd2_it = previous_upd2->begin();
					std::vector<double>::iterator updates_accumulated_it = updates_accumulated.begin();
					std::vector<float>::const_iterator learning_rate_it = learning_rates.begin();
					std::set<unsigned int> weight_decay_part_id_set = schema->get_layer(layer_name)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					float one_minus_beta1t_inverted = 1.0F / (1.0F - powf(momentum.momentum_val, static_cast<float>(iteration_id)));
					float one_minus_beta2t_inverted = 1.0F / (1.0F - powf(momentum.momentum_val2, static_cast<float>(iteration_id)));
					float epsilon = 1.0e-8F;
					for(layer_data::iterator data_it = data->begin(); data_it != data->end(); ++data_it, ++gradient_it, ++previous_upd_it, ++previous_upd2_it, ++learning_rate_it, ++part_id, ++updates_accumulated_it)
					{
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						std::vector<float>::iterator previous_upd_it2 = previous_upd_it->begin();
						std::vector<float>::iterator previous_upd2_it2 = previous_upd2_it->begin();
						float learning_rate = *learning_rate_it;
						double accum = 0.0;
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++previous_upd_it2, ++previous_upd2_it2)
						{
							float current_weight = *data_it2;
							float gr = *gradient_it2;
							float previous_biased_first_momentum = *previous_upd_it2;
							float previous_biased_second_momentum = *previous_upd2_it2;
							float total_gradient = gr * normalizer - current_weight * weight_decay;
							float new_biased_first_momentum = momentum.momentum_val * previous_biased_first_momentum + (1.0F - momentum.momentum_val) * total_gradient;
							float new_biased_second_momentum = momentum.momentum_val2 * previous_biased_second_momentum + (1.0F - momentum.momentum_val2) * total_gradient * total_gradient;
							float unbiased_first_momentum = new_biased_first_momentum * one_minus_beta1t_inverted;
							float unbiased_second_momentum = new_biased_second_momentum * one_minus_beta2t_inverted;
							float upd = (learning_rate * unbiased_first_momentum) / (sqrtf(unbiased_second_momentum) + epsilon);
							float new_weight = current_weight + upd;
							accum += static_cast<double>(fabsf(upd));
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
							*previous_upd_it2 = new_biased_first_momentum;
							*previous_upd2_it2 = new_biased_second_momentum;
						}
						*updates_accumulated_it += accum;
					}
				}
				break;
			}
		}
	}
}
