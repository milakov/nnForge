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
#include <numeric>

#include <boost/format.hpp>

#include "layer_tester_plain_factory.h"
#include "layer_updater_plain_factory.h"

#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "../debug_util.h"
#include <boost/filesystem.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nnforge
{
	namespace plain
	{
		unsigned int network_updater_plain::max_entry_count_in_single_batch = 1024;

		network_updater_plain::network_updater_plain(
			network_schema_smart_ptr schema,
			const_error_function_smart_ptr ef,
			const std::map<unsigned int, float>& layer_to_dropout_rate_map,
			plain_running_configuration_const_smart_ptr plain_config)
			: network_updater(schema, ef, layer_to_dropout_rate_map)
			, plain_config(plain_config)
		{
			const const_layer_list& layer_list = *schema;

			error_function_fused_with_activation = (layer_list.back()->get_uuid() == ef->get_fusable_activation_uuid());

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
			{
				if ((it != layer_list.end() - 1) || (!error_function_fused_with_activation))
					updater_list.push_back(single_layer_updater_plain_factory::get_const_instance().get_updater_plain_layer((*it)->get_uuid()));
			}
		}

		network_updater_plain::~network_updater_plain()
		{
		}

		testing_result_smart_ptr network_updater_plain::actual_update(
			supervised_data_reader& reader,
			network_data_const_smart_ptr learning_rate,
			network_data_smart_ptr data,
			unsigned int batch_size,
			float weight_decay,
			float momentum)
		{
			testing_result_smart_ptr res(new testing_result(ef));

			reader.reset();

			const unsigned int input_neuron_count = reader.get_input_configuration().get_neuron_count();
			const unsigned int output_neuron_count = reader.get_output_configuration().get_neuron_count();
			const unsigned int input_feature_map_count = reader.get_input_configuration().feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = reader.get_input_configuration().get_neuron_count_per_feature_map();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			unsigned int updater_max_count = std::max(get_updater_max_count(), 1U);
			unsigned int updater_entry_count;
			std::vector<unsigned int> entry_read_count_list;
			unsigned int max_entry_read_count;
			if (updater_max_count > batch_size)
				updater_entry_count = batch_size;
			else
			{
				unsigned int it_count = (batch_size + updater_max_count - 1) / updater_max_count;
				updater_entry_count = (batch_size + it_count - 1) / it_count;
				max_entry_read_count = updater_entry_count;
				unsigned int sum = 0;
				while (sum < batch_size)
				{
					unsigned int new_item = std::min(batch_size - sum, updater_entry_count);
					sum += new_item;
					entry_read_count_list.push_back(new_item);
				}
			}

			network_data_smart_ptr gradient(new network_data(*schema));
			gradient->fill(0.0F);
			network_data_smart_ptr previous_upd;
			if (momentum > 0.0F)
			{
				previous_upd = network_data_smart_ptr(new network_data(*schema));
				previous_upd->fill(0.0F);
			}

			{
				buffer_plain_size_configuration buffers_config;
				update_buffers_configuration(buffers_config, updater_entry_count);
				buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
				buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
				buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
				buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // initial error
				for(std::vector<layer_data_smart_ptr>::iterator it = data->begin(); it != data->end(); ++it)
				{
					for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
					{
						buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // data
						buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // learning rate
						buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // gradient
					}
				}

				unsigned int max_entry_count = std::min(std::min(plain_config->get_max_entry_count(buffers_config), reader.get_entry_count()), max_entry_count_in_single_batch);
				if (entry_read_count_list.empty() || (max_entry_count >= batch_size))
				{
					unsigned int it_count = std::max((max_entry_count + batch_size - 1) / batch_size, 1U);
					max_entry_read_count = it_count * batch_size;
					entry_read_count_list.clear();
					entry_read_count_list.push_back(max_entry_read_count);
				}
			}

			std::vector<unsigned char> input_buf(max_entry_read_count * input_neuron_count * input_neuron_elem_size);
			std::vector<float> actual_output_buf(max_entry_read_count * output_neuron_count);
			additional_buffer_smart_ptr initial_error_buf(new std::vector<float>(updater_entry_count * output_neuron_count));
			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count * max_entry_read_count));

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
						max_entry_read_count,
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
			unsigned int entry_read_count_index = 0;
			unsigned int entry_gradient_calculated_count = 0;
			while (entries_remained_for_loading)
			{
				unsigned int entries_available_for_processing_count = 0;
				while(entries_available_for_processing_count < entry_read_count_list[entry_read_count_index])
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
				entry_read_count_index++;
				if (entry_read_count_index >= entry_read_count_list.size())
					entry_read_count_index = 0;

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

				unsigned int base_input_entry_id = 0;
				while(base_input_entry_id < entries_available_for_processing_count)
				{
					std::stack<unsigned int> offset_list;

					unsigned int current_updater_entry_count = std::min(std::min(entries_available_for_processing_count - base_input_entry_id, updater_entry_count), batch_size - entry_gradient_calculated_count);

					// Forward updater
					{
						const_layer_list::const_iterator layer_it = layer_list.begin() + testing_layer_count;
						layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin() + testing_layer_count;
						std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.begin();
						layer_data_list::const_iterator data_it = data->begin() + testing_layer_count;
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
										current_updater_entry_count * layer_config_list[layer_id].get_neuron_count(),
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
								current_updater_entry_count,
								(it == updater_list.begin()) ? base_input_entry_id : 0);
						}
					}

					// Set initial error and accumulate error
					{
						const std::vector<float>::iterator initial_error_it = initial_error_buf->begin();
						const std::vector<float>::const_iterator actual_output_buf_it = actual_output_buf.begin() + (output_neuron_count * base_input_entry_id);
						const std::vector<float>::const_iterator output_buffer_it = output_buffer->begin();
						testing_result& tr = *res;
						const int elem_count = current_updater_entry_count;
						std::vector<double> errors(plain_config->openmp_thread_count, 0.0);
						const std::vector<double>::iterator errors_it = errors.begin();
						#pragma omp parallel default(none) shared(tr) num_threads(plain_config->openmp_thread_count)
						{
							int thread_id = 0;
							#ifdef _OPENMP
								thread_id = omp_get_thread_num();
							#endif

							#pragma omp for schedule(guided)
							for(int updater_entry_id = 0; updater_entry_id < elem_count; ++updater_entry_id)
							{
								const float * predicted_vals = &(*(output_buffer_it + (updater_entry_id * output_neuron_count)));
								const float * actual_vals = &(*(actual_output_buf_it + (updater_entry_id * output_neuron_count)));
								float * initial_errors = &(*(initial_error_it + (updater_entry_id * output_neuron_count)));

								float error;
								if (error_function_fused_with_activation)
									error = tr.ef->calculate_gradient_and_error_fused_with_activation(actual_vals, predicted_vals, initial_errors, output_neuron_count);
								else
									error = tr.ef->calculate_gradient_and_error(actual_vals, predicted_vals, initial_errors, output_neuron_count);
								*(errors_it + thread_id) += static_cast<double>(error);
							}
						}
						double total_error = std::accumulate(errors.begin(), errors.end(), 0.0);
						tr.add_error(total_error, current_updater_entry_count);
					}

					// Run backward and update weights
					{
						const_layer_list::const_reverse_iterator layer_it = layer_list.rbegin() + (error_function_fused_with_activation ? 1 : 0);
						std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::reverse_iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.rbegin();
						layer_configuration_specific_list::const_reverse_iterator input_config_it = layer_config_list.rbegin() + (error_function_fused_with_activation ? 1 : 0);
						layer_data_list::const_reverse_iterator data_it = data->rbegin() + (error_function_fused_with_activation ? 1 : 0);
						layer_data_list::reverse_iterator gradient_it = gradient->rbegin() + (error_function_fused_with_activation ? 1 : 0);
						additional_buffer_smart_ptr output_errors = initial_error_buf;
						unsigned int reverse_layer_id = static_cast<unsigned int>(updater_list.size() + testing_layer_count) - 1 - (error_function_fused_with_activation ? 1 : 0);
						for(std::vector<const_layer_updater_plain_smart_ptr>::const_reverse_iterator it = updater_list.rbegin(); it != updater_list.rend(); ++it, ++layer_it, ++input_config_it, ++updater_buffers_it, ++data_it, ++gradient_it, --reverse_layer_id)
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
									current_updater_entry_count);
								
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
										current_updater_entry_count * layer_config_list[reverse_layer_id].get_neuron_count(),
										offset);
								}
							}

							(*it)->update_weights(
								updater_buffers_it->first,
								output_errors,
								updater_buffers_it->second.additional_buffers,
								*gradient_it,
								plain_config,
								*layer_it,
								*(input_config_it + 1),
								*input_config_it,
								current_updater_entry_count,
								(it == updater_list.rend() - 1) ? base_input_entry_id : 0);

							output_errors = updater_buffers_it->second.input_errors_buffer;
						}
					}

					base_input_entry_id += current_updater_entry_count;
					entry_gradient_calculated_count += current_updater_entry_count;

					if (entry_gradient_calculated_count >= batch_size)
					{
						float gradient_normalizer = 1.0F / static_cast<float>(std::max(batch_size, entry_gradient_calculated_count));
						apply_gradient(
							*data,
							*gradient,
							*previous_upd,
							*learning_rate,
							gradient_normalizer,
							weight_decay,
							momentum);
						entry_gradient_calculated_count = 0;
					}
				}
			}

			if (entry_gradient_calculated_count > 0)
			{
				float gradient_normalizer = 1.0F / static_cast<float>(std::max(batch_size, entry_gradient_calculated_count));
				apply_gradient(
					*data,
					*gradient,
					*previous_upd,
					*learning_rate,
					gradient_normalizer,
					weight_decay,
					momentum);
				entry_gradient_calculated_count = 0;
			}

			return res;
		}

		void network_updater_plain::layer_config_list_modified()
		{
		}

		void network_updater_plain::apply_gradient(
			std::vector<layer_data_smart_ptr>& data,
			std::vector<layer_data_smart_ptr>& gradient,
			std::vector<layer_data_smart_ptr>& previous_upd,
			const std::vector<layer_data_smart_ptr>& learning_rate,
			float normalizer,
			float weight_decay,
			float momentum) const
		{
			if (momentum > 0.0F)
			{
				layer_data_list::iterator gradient_it0 = gradient.begin() + testing_layer_count;
				layer_data_list::iterator previous_upd_it0 = previous_upd.begin() + testing_layer_count;
				layer_data_list::const_iterator learning_rate_it0 = learning_rate.begin() + testing_layer_count;
				for(layer_data_list::iterator data_it0 = data.begin() + testing_layer_count; data_it0 != data.end(); ++data_it0, ++gradient_it0, ++previous_upd_it0, ++learning_rate_it0)
				{
					layer_data::iterator gradient_it = (*gradient_it0)->begin();
					layer_data::iterator previous_upd_it = (*previous_upd_it0)->begin();
					layer_data::const_iterator learning_rate_it = (*learning_rate_it0)->begin();
					for(layer_data::iterator data_it = (*data_it0)->begin(); data_it != (*data_it0)->end(); ++data_it, ++gradient_it, ++previous_upd_it, ++learning_rate_it)
					{
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						std::vector<float>::iterator previous_upd_it2 = previous_upd_it->begin();
						std::vector<float>::const_iterator learning_rate_it2 = learning_rate_it->begin();
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++previous_upd_it2, ++learning_rate_it2)
						{
							float current_weight = *data_it2;
							float lr = *learning_rate_it2;
							float gr = *gradient_it2;
							float prev_upd = *previous_upd_it2;
							float upd = prev_upd * momentum + lr * (gr * normalizer - current_weight * weight_decay);
							float new_weight = current_weight + upd;
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
							*previous_upd_it2 = upd;
						}
					}
				}
			}
			else
			{
				layer_data_list::iterator gradient_it0 = gradient.begin() + testing_layer_count;
				layer_data_list::const_iterator learning_rate_it0 = learning_rate.begin() + testing_layer_count;
				for(layer_data_list::iterator data_it0 = data.begin() + testing_layer_count; data_it0 != data.end(); ++data_it0, ++gradient_it0, ++learning_rate_it0)
				{
					layer_data::iterator gradient_it = (*gradient_it0)->begin();
					layer_data::const_iterator learning_rate_it = (*learning_rate_it0)->begin();
					for(layer_data::iterator data_it = (*data_it0)->begin(); data_it != (*data_it0)->end(); ++data_it, ++gradient_it, ++learning_rate_it)
					{
						std::vector<float>::iterator gradient_it2 = gradient_it->begin();
						std::vector<float>::const_iterator learning_rate_it2 = learning_rate_it->begin();
						for(std::vector<float>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++learning_rate_it2)
						{
							float current_weight = *data_it2;
							float lr = *learning_rate_it2;
							float gr = *gradient_it2;
							float new_weight = current_weight + lr * (gr * normalizer - current_weight * weight_decay);
							*data_it2 = new_weight;
							*gradient_it2 = 0.0F;
						}
					}
				}
			}
		}

		unsigned int network_updater_plain::get_updater_max_count() const
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
