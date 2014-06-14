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

#include "network_updater_plain.h"

#include <stack>

#include <boost/format.hpp>

#include "layer_tester_plain_factory.h"
#include "layer_updater_plain_factory.h"
#include "weight_vector_bound_plain_factory.h"

#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "../debug_util.h"
#include <boost/filesystem.hpp>

namespace nnforge
{
	namespace plain
	{
		unsigned int network_updater_plain::max_entry_count_in_single_batch = 1024;

		network_updater_plain::network_updater_plain(
			network_schema_smart_ptr schema,
			const_error_function_smart_ptr ef,
			const std::map<unsigned int, float>& layer_to_dropout_rate_map,
			const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
			float weight_decay,
			plain_running_configuration_const_smart_ptr plain_config)
			: network_updater(schema, ef, layer_to_dropout_rate_map, layer_to_weight_vector_bound_map, weight_decay)
			, plain_config(plain_config)
		{
			const const_layer_list& layer_list = *schema;

			testing_layer_count = 0;
			start_layer_nonempty_weights_iterator = layer_list.begin();
			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
			{
				start_layer_nonempty_weights_iterator = it;

				if (!(*it)->is_empty_data())
					break;

				testing_layer_count++;
			}

			for(const_layer_list::const_iterator it = layer_list.begin(); it != start_layer_nonempty_weights_iterator; ++it)
				tester_list.push_back(single_layer_tester_plain_factory::get_const_instance().get_tester_plain_layer((*it)->get_uuid()));

			for(const_layer_list::const_iterator it = start_layer_nonempty_weights_iterator; it != layer_list.end(); ++it)
				updater_list.push_back(single_layer_updater_plain_factory::get_const_instance().get_updater_plain_layer((*it)->get_uuid()));

			for(std::map<unsigned int, weight_vector_bound>::const_iterator it = this->layer_to_weight_vector_bound_map.begin(); it != this->layer_to_weight_vector_bound_map.end(); ++it)
			{
				unsigned int layer_id = it->first;
				if (layer_id < testing_layer_count)
					throw neural_network_exception((boost::format("Weight vector bound is specified for layer %1% while it is in testing part (consisting of %2% layers) of the updater") % layer_id  % testing_layer_count).str());

				weight_vector_bounds.insert(std::make_pair(layer_id, single_weight_vector_bound_factory::get_const_instance().get_updater_plain_layer(layer_list[layer_id]->get_uuid())));
			}
		}

		network_updater_plain::~network_updater_plain()
		{
		}

		std::vector<testing_result_smart_ptr> network_updater_plain::actual_update(
			supervised_data_reader& reader,
			const std::vector<network_data_smart_ptr>& learning_rate_vector_list,
			std::vector<network_data_smart_ptr>& data_list)
		{
			std::vector<testing_result_smart_ptr> res;

			const unsigned int input_neuron_count = reader.get_input_configuration().get_neuron_count();
			const unsigned int output_neuron_count = reader.get_output_configuration().get_neuron_count();
			const unsigned int input_feature_map_count = reader.get_input_configuration().feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = reader.get_input_configuration().get_neuron_count_per_feature_map();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			unsigned int updater_entry_count = static_cast<unsigned int>(data_list.size());

			if (updater_entry_count == 0)
				return res;

			for(unsigned int i = 0; i < learning_rate_vector_list.size(); ++i)
				res.push_back(testing_result_smart_ptr(new testing_result(ef)));

			buffer_plain_size_configuration buffers_config;
			update_buffers_configuration(buffers_config, updater_entry_count);
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // initial error
			for(std::vector<network_data_smart_ptr>::iterator it3 = data_list.begin(); it3 != data_list.end(); ++it3)
			{
				for(std::vector<layer_data_smart_ptr>::iterator it = (*it3)->begin(); it != (*it3)->end(); ++it)
				{
					for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
					{
						buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // data
						buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // training speed
					}
				}
			}

			std::vector<layer_data_list> data_list_reorganized(data_list[0]->size() - testing_layer_count);
			for(unsigned int layer_id = testing_layer_count; layer_id < data_list[0]->size(); ++layer_id)
				for(unsigned int updater_entry_id = 0; updater_entry_id < updater_entry_count; ++updater_entry_id)
					data_list_reorganized[layer_id - testing_layer_count].push_back((*data_list[updater_entry_id])[layer_id]);

			std::vector<layer_data_list> learning_rate_vector_list_reorganized(learning_rate_vector_list[0]->size() - testing_layer_count);
			for(unsigned int layer_id = testing_layer_count; layer_id < learning_rate_vector_list[0]->size(); ++layer_id)
				for(unsigned int updater_entry_id = 0; updater_entry_id < updater_entry_count; ++updater_entry_id)
					learning_rate_vector_list_reorganized[layer_id - testing_layer_count].push_back((*learning_rate_vector_list[updater_entry_id])[layer_id]);

			unsigned int max_entry_count = std::min<unsigned int>(std::min<unsigned int>(plain_config->get_max_entry_count(buffers_config), reader.get_entry_count()), max_entry_count_in_single_batch);

			std::vector<unsigned char> input_buf(max_entry_count * input_neuron_count * input_neuron_elem_size);
			std::vector<float> actual_output_buf(max_entry_count * output_neuron_count);
			additional_buffer_smart_ptr initial_error_buf(new std::vector<float>(updater_entry_count * output_neuron_count));
			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count * max_entry_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_testing_buffers_pack;
			std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> > input_buffer_and_additional_updater_buffers_pack;
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_iterator layer_it = layer_list.begin();
				layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
				for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it)
				{
					additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
						max_entry_count,
						*layer_it,
						*input_config_it,
						*(input_config_it + 1),
						plain_config);
					input_buffer_and_additional_testing_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				}
				for(const_layer_updater_plain_list::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it)
				{
					updater_additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
						updater_entry_count,
						*layer_it,
						*input_config_it,
						*(input_config_it + 1),
						plain_config,
						(it != updater_list.begin()));
					input_buffer_and_additional_updater_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
					output_buffer = additional_buffers.output_neurons_buffer;
				}
			}
			{
				additional_buffer_smart_ptr output_errors = initial_error_buf;
				for(std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::reverse_iterator it = input_buffer_and_additional_updater_buffers_pack.rbegin(); it != input_buffer_and_additional_updater_buffers_pack.rend() - 1; ++it)
				{
					if (it->second.input_errors_buffer != 0)
						output_errors = it->second.input_errors_buffer;
					else
						it->second.input_errors_buffer = output_errors;
				}
			}

			random_generator gen = rnd::get_random_generator();
			nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(random_uniform_list.size() - 1));
			unsigned int mask = static_cast<unsigned int>(random_uniform_list.size() - 1);
			bool entries_remained_for_loading = true;
			while (entries_remained_for_loading)
			{
				unsigned int entries_available_for_processing_count = 0;
				while(entries_available_for_processing_count < max_entry_count)
				{
					bool entry_read = reader.read(
						&(*(input_buf.begin() + (input_neuron_count * entries_available_for_processing_count * input_neuron_elem_size))),
						&(*(actual_output_buf.begin() + (output_neuron_count * entries_available_for_processing_count))));
					if (!entry_read)
					{
						entries_remained_for_loading = false;
						break;
					}
					entries_available_for_processing_count++;
				}

				if (entries_available_for_processing_count == 0)
					break;

				const unsigned int const_entries_available_for_processing_count = entries_available_for_processing_count;

				// Convert input
				{
					const int elem_count = static_cast<int>(entries_available_for_processing_count * input_neuron_count);
					const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
					if (type_code == neuron_data_type::type_byte)
					{
						const unsigned char * const input_buf_it_start = &(*input_buf.begin());
						#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
						for(int i = 0; i < elem_count; ++i)
							*(input_converted_buf_it_start + i) = static_cast<float>(*(input_buf_it_start + i)) * (1.0F / 255.0F);
					}
					else if (type_code == neuron_data_type::type_float)
					{
						const float * const input_buf_it_start = reinterpret_cast<float *>(&(*input_buf.begin()));
						#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
						for(int i = 0; i < elem_count; ++i)
							*(input_converted_buf_it_start + i) = *(input_buf_it_start + i);
					}
					else
						throw neural_network_exception((boost::format("actual_update cannot handle input neurons of type %1%") % type_code).str());
				}

				// Run testing layers
				const const_layer_list& layer_list = *schema;
				{
					const_layer_list::const_iterator layer_it = layer_list.begin();
					unsigned int layer_id = 0;
					layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
					std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_testing_buffers_pack.begin();
					for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it, ++buffers_it, ++layer_id)
					{
						std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
						if (dropout_it != layer_to_dropout_rate_map.end())
						{
							unsigned int offset = dist(gen);
							apply_dropout(
								buffers_it->first,
								dropout_it->second,
								mask,
								entries_available_for_processing_count * layer_config_list[layer_id].get_neuron_count(),
								offset);
						}

						(*it)->test(
							buffers_it->first,
							buffers_it->second,
							plain_config,
							*layer_it,
							const_layer_data_smart_ptr(),
							*input_config_it,
							*(input_config_it + 1),
							entries_available_for_processing_count);
					}
				}

				// Apply dropout to the input of the first updater layer
				{
					std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(testing_layer_count);
					if (dropout_it != layer_to_dropout_rate_map.end())
					{
						unsigned int offset = dist(gen);
						apply_dropout(
							input_buffer_and_additional_updater_buffers_pack[0].first,
							dropout_it->second,
							mask,
							entries_available_for_processing_count * layer_config_list[testing_layer_count].get_neuron_count(),
							offset);
					}
				}

				for(unsigned int input_entry_id = 0; input_entry_id < entries_available_for_processing_count; ++input_entry_id)
				{
					std::stack<unsigned int> offset_list;

					// Forward updater
					{
						const_layer_list::const_iterator layer_it = layer_list.begin() + testing_layer_count;
						layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin() + testing_layer_count;
						std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.begin();
						std::vector<layer_data_list>::const_iterator data_it = data_list_reorganized.begin();
						unsigned int layer_id = testing_layer_count;
						for(std::vector<const_layer_updater_plain_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it, ++updater_buffers_it, ++data_it, ++layer_id)
						{
							if (it != updater_list.begin())
							{
								std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
								if (dropout_it != layer_to_dropout_rate_map.end())
								{
									unsigned int offset = dist(gen);
									offset_list.push(offset);
									apply_dropout(
										updater_buffers_it->first,
										dropout_it->second,
										mask,
										updater_entry_count * layer_config_list[layer_id].get_neuron_count(),
										offset);
								}
							}

							(*it)->test(
								updater_buffers_it->first,
								updater_buffers_it->second.output_neurons_buffer,
								updater_buffers_it->second.additional_buffers,
								plain_config,
								*layer_it,
								*data_it,
								*input_config_it,
								*(input_config_it + 1),
								updater_entry_count,
								(it == updater_list.begin()) ? input_entry_id : -1);
						}
					}

					// Set initial error and compute temporary MSE
					{
						const std::vector<float>::iterator initial_error_it = initial_error_buf->begin();
						const std::vector<float>::const_iterator actual_output_buf_it = actual_output_buf.begin() + (output_neuron_count * input_entry_id);
						const std::vector<float>::const_iterator output_buffer_it = output_buffer->begin();
						const std::vector<testing_result_smart_ptr>::iterator testing_res_it = res.begin();
						const int elem_count = updater_entry_count;
						#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
						for(int updater_entry_id = 0; updater_entry_id < elem_count; ++updater_entry_id)
						{
							const float * predicted_vals = &(*(output_buffer_it + (updater_entry_id * output_neuron_count)));
							const float * actual_vals = &(*actual_output_buf_it);
							float * initial_errors = &(*(initial_error_it + (updater_entry_id * output_neuron_count)));
							testing_result& tr = **(testing_res_it + updater_entry_id);

							tr.add_error(actual_vals, predicted_vals, output_neuron_count);
							tr.ef->calculate_gradient(actual_vals, predicted_vals, initial_errors, output_neuron_count);
						}
					}

					// Run backward and update weights
					{
						const_layer_list::const_reverse_iterator layer_it = layer_list.rbegin();
						std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::reverse_iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.rbegin();
						layer_configuration_specific_list::const_reverse_iterator input_config_it = layer_config_list.rbegin();
						std::vector<layer_data_list>::reverse_iterator data_it = data_list_reorganized.rbegin();
						std::vector<layer_data_list>::const_reverse_iterator learning_rate_it = learning_rate_vector_list_reorganized.rbegin();
						additional_buffer_smart_ptr output_errors = initial_error_buf;
						unsigned int reverse_layer_id = static_cast<unsigned int>(updater_list.size() + testing_layer_count) - 1;
						for(std::vector<const_layer_updater_plain_smart_ptr>::const_reverse_iterator it = updater_list.rbegin(); it != updater_list.rend(); ++it, ++layer_it, ++input_config_it, ++updater_buffers_it, ++data_it, ++learning_rate_it, --reverse_layer_id)
						{
							if (it != updater_list.rend() - 1)
							{
								(*it)->backprop(
									updater_buffers_it->second.input_errors_buffer,
									updater_buffers_it->first,
									output_errors,
									updater_buffers_it->second.output_neurons_buffer,
									updater_buffers_it->second.additional_buffers,
									plain_config,
									*layer_it,
									*data_it,
									*(input_config_it + 1),
									*input_config_it,
									updater_entry_count);
								/*
								{
									boost::filesystem::path dir = "Debug";
									dir /= "CPU";
									boost::filesystem::create_directories(dir);
									debug_util::dump_list(
										&(*updater_buffers_it->second.input_errors_buffer->begin()),
										updater_buffers_it->second.input_errors_buffer->size(),
										(dir / (boost::format("input_errors_%1%.txt") % reverse_layer_id).str()).string().c_str());
								}
								*/

								std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(reverse_layer_id);
								if (dropout_it != layer_to_dropout_rate_map.end())
								{
									unsigned int offset = offset_list.top();
									offset_list.pop();
									apply_dropout(
										updater_buffers_it->second.input_errors_buffer,
										dropout_it->second,
										mask,
										updater_entry_count * layer_config_list[reverse_layer_id].get_neuron_count(),
										offset);
								}
							}

							(*it)->update_weights(
								updater_buffers_it->first,
								output_errors,
								updater_buffers_it->second.additional_buffers,
								*data_it,
								*learning_rate_it,
								plain_config,
								*layer_it,
								*(input_config_it + 1),
								*input_config_it,
								updater_entry_count,
								(it == updater_list.rend() - 1) ? input_entry_id : -1,
								weight_decay);

							weight_vector_bound_map::iterator bound_it = weight_vector_bounds.find(reverse_layer_id);
							if (bound_it != weight_vector_bounds.end())
							{
								const weight_vector_bound& bound = layer_to_weight_vector_bound_map.find(reverse_layer_id)->second;
								bound_it->second->normalize_weights(
									bound,
									*data_it,
									plain_config,
									*layer_it,
									updater_entry_count);
							}

							output_errors = updater_buffers_it->second.input_errors_buffer;
						}
					}
				}
			}

			return res;
		}

		void network_updater_plain::layer_config_list_modified()
		{
		}

		unsigned int network_updater_plain::get_max_batch_size() const
		{
			buffer_plain_size_configuration buffer_configuration;

			const const_layer_list& layer_list = *schema;
			const_layer_list::const_iterator layer_it = layer_list.begin();
			layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
			for(const_layer_updater_plain_list::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				(*it)->update_buffer_configuration(
					buffer_configuration,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config,
					(it != updater_list.begin()));
			}

			return plain_config->get_max_entry_count(buffer_configuration, 0.5F);
		}

		void network_updater_plain::update_buffers_configuration(
			buffer_plain_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			const const_layer_list& layer_list = *schema;
			const_layer_list::const_iterator layer_it = layer_list.begin();
			layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
			for(const_layer_tester_plain_list::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				(*it)->update_buffer_configuration(
					buffer_configuration,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config);
			}
			for(const_layer_updater_plain_list::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				(*it)->update_buffer_configuration(
					buffer_configuration,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config,
					(it != updater_list.begin()),
					updater_entry_count);
			}
		}

		void network_updater_plain::apply_dropout(
			additional_buffer_smart_ptr target_buffer,
			const float dropout_rate,
			const unsigned int mask,
			const unsigned int elem_count,
			const unsigned int offset_in_random_list) const
		{
			const std::vector<float>::const_iterator rnd_it = random_uniform_list.begin();
			const std::vector<float>::iterator in_it = target_buffer->begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
			{
				unsigned int random_elem_id = (i + offset_in_random_list) & mask;
				if (*(rnd_it + random_elem_id) < dropout_rate)
					*(in_it + i) = 0.0F;
			}
		}
	}
}
