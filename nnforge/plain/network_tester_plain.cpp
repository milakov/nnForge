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

#include "network_tester_plain.h"

#include "layer_tester_plain_factory.h"

namespace nnforge
{
	namespace plain
	{
		network_tester_plain::network_tester_plain(
			network_schema_smart_ptr schema,
			const_data_scale_params_smart_ptr scale_params,
			plain_running_configuration_const_smart_ptr plain_config)
			: network_tester(schema, scale_params)
			, plain_config(plain_config)
		{
			const const_layer_list& layer_list = *schema;
			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				tester_list.push_back(plain::single_layer_tester_plain_factory::get_const_instance().get_tester_plain_layer((*it)->get_uuid()));
		}

		network_tester_plain::~network_tester_plain()
		{
		}

		void network_tester_plain::actual_test(
			supervised_data_reader_byte& reader,
			testing_complete_result_set& result)
		{
			reader.reset();

			const unsigned int input_neuron_count = reader.get_input_configuration().get_neuron_count();
			const unsigned int output_neuron_count = reader.get_output_configuration().get_neuron_count();
			const unsigned int entry_count = reader.get_entry_count();
			const unsigned int input_feature_map_count = reader.get_input_configuration().feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = reader.get_input_configuration().get_neuron_count_per_feature_map();
			result.mse = testing_result_smart_ptr(new testing_result(output_neuron_count));

			buffer_plain_size_configuration buffers_config;
			update_buffers_configuration_testing(buffers_config);
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input

			const unsigned int max_entry_count = std::min<unsigned int>(plain_config->get_max_entry_count(buffers_config), reader.get_entry_count());

			std::vector<unsigned char> input_buf(input_neuron_count * max_entry_count);
			std::vector<float> actual_output_buf(output_neuron_count * max_entry_count);
			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count * max_entry_count));
			std::vector<float>& mse_buf = result.mse->cumulative_mse_list;

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_buffers_pack;
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
					input_buffer_and_additional_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				}
			}

			bool entries_remained_for_loading = true;
			while (entries_remained_for_loading)
			{
				unsigned int entries_available_for_processing_count = 0;
				while(entries_available_for_processing_count < max_entry_count)
				{
					bool entry_read = reader.read(
						&(*(input_buf.begin() + (input_neuron_count * entries_available_for_processing_count))),
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
					const int elem_count = static_cast<int>(const_entries_available_for_processing_count);
					const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
					const std::vector<unsigned char>::const_iterator input_buf_it_start = input_buf.begin();
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < elem_count; ++i)
					{
						std::vector<float>::iterator input_converted_buf_it = input_converted_buf_it_start + (i * input_neuron_count);
						std::vector<unsigned char>::const_iterator input_buf_it = input_buf_it_start + (i * input_neuron_count);
						for(unsigned int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
						{
							float addition = current_scale_params->addition_list[feature_map_id];
							float multiplication = current_scale_params->multiplication_list[feature_map_id];
							for(unsigned int j = 0; j < neuron_count_per_input_feature_map; ++j)
							{
								*input_converted_buf_it = ((static_cast<float>(*input_buf_it) * (1.0F / 255.0F)) + addition) * multiplication;
								input_converted_buf_it++;
								input_buf_it++;
							}
						}
					}
				}

				// Run ann
				{
					const const_layer_list& layer_list = *schema;
					const_layer_list::const_iterator layer_it = layer_list.begin();
					layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
					std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_buffers_pack.begin();
					layer_data_list::const_iterator data_it = net_data->begin();
					for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it, ++buffers_it, ++data_it)
					{
						(*it)->test(
							buffers_it->first,
							buffers_it->second,
							plain_config,
							*layer_it,
							*data_it,
							*input_config_it,
							*(input_config_it + 1),
							entries_available_for_processing_count);
					}
				}

				// Compute MSE
				{
					const int total_workload = static_cast<int>(output_neuron_count);
					const std::vector<float>::iterator mse_buf_it = mse_buf.begin();
					const std::vector<float>::const_iterator actual_output_buf_it = actual_output_buf.begin();
					const std::vector<float>::const_iterator output_buffer_it = output_buffer->begin();
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < total_workload; ++i)
					{
						float mse_local2 = 0.0F;
						unsigned int elem_id = i;
						for(unsigned int j = 0; j < const_entries_available_for_processing_count; ++j)
						{
							float diff = *(actual_output_buf_it + elem_id) - *(output_buffer_it + elem_id);
							mse_local2 += diff * diff;
							elem_id += total_workload;
						}
						*(mse_buf_it + i) += mse_local2 * 0.5F;
					}
				}

				// Copy predicted values
				{
					const int total_workload = static_cast<int>(entries_available_for_processing_count);
					const std::vector<float>::const_iterator output_buffer_it = output_buffer->begin();
					const std::vector<std::vector<float> >::iterator neuron_value_list_it = result.predicted_output_neuron_value_set->neuron_value_list.begin() + result.mse->entry_count;
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < total_workload; ++i)
					{
						std::vector<float>::const_iterator src_it = output_buffer_it + (i * output_neuron_count);
						std::vector<float>& value_list_dest = *(neuron_value_list_it + i);
						std::copy(src_it, src_it + output_neuron_count, value_list_dest.begin());
					}
				}

				result.mse->entry_count += entries_available_for_processing_count;
			}
		}

		output_neuron_value_set_smart_ptr network_tester_plain::actual_run(unsupervised_data_reader_byte& reader)
		{
			reader.reset();

			const unsigned int input_neuron_count = reader.get_input_configuration().get_neuron_count();
			const unsigned int output_neuron_count = layer_config_list[layer_config_list.size() - 1].get_neuron_count();
			const unsigned int entry_count = reader.get_entry_count();
			const unsigned int input_feature_map_count = reader.get_input_configuration().feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = reader.get_input_configuration().get_neuron_count_per_feature_map();

			output_neuron_value_set_smart_ptr predicted_output_neuron_value_set(new output_neuron_value_set(entry_count, output_neuron_count));

			buffer_plain_size_configuration buffers_config;
			update_buffers_configuration_testing(buffers_config);
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input

			const unsigned int max_entry_count = std::min<unsigned int>(plain_config->get_max_entry_count(buffers_config), reader.get_entry_count());

			std::vector<unsigned char> input_buf(input_neuron_count * max_entry_count);
			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count * max_entry_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_buffers_pack;
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
					input_buffer_and_additional_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				}
			}

			bool entries_remained_for_loading = true;
			unsigned int entries_copied_count = 0;
			while (entries_remained_for_loading)
			{
				unsigned int entries_available_for_processing_count = 0;
				while(entries_available_for_processing_count < max_entry_count)
				{
					bool entry_read = reader.read(&(*(input_buf.begin() + (input_neuron_count * entries_available_for_processing_count))));
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
					const int elem_count = static_cast<int>(const_entries_available_for_processing_count);
					const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
					const std::vector<unsigned char>::const_iterator input_buf_it_start = input_buf.begin();
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < elem_count; ++i)
					{
						std::vector<float>::iterator input_converted_buf_it = input_converted_buf_it_start + (i * input_neuron_count);
						std::vector<unsigned char>::const_iterator input_buf_it = input_buf_it_start + (i * input_neuron_count);
						for(unsigned int feature_map_id = 0; feature_map_id < input_feature_map_count; ++feature_map_id)
						{
							float addition = current_scale_params->addition_list[feature_map_id];
							float multiplication = current_scale_params->multiplication_list[feature_map_id];
							for(unsigned int j = 0; j < neuron_count_per_input_feature_map; ++j)
							{
								*input_converted_buf_it = ((static_cast<float>(*input_buf_it) * (1.0F / 255.0F)) + addition) * multiplication;
								input_converted_buf_it++;
								input_buf_it++;
							}
						}
					}
				}

				// Run ann
				{
					const const_layer_list& layer_list = *schema;
					const_layer_list::const_iterator layer_it = layer_list.begin();
					layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
					std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_buffers_pack.begin();
					layer_data_list::const_iterator data_it = net_data->begin();
					for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it, ++buffers_it, ++data_it)
					{
						(*it)->test(
							buffers_it->first,
							buffers_it->second,
							plain_config,
							*layer_it,
							*data_it,
							*input_config_it,
							*(input_config_it + 1),
							entries_available_for_processing_count);
					}
				}

				// Copy predicted values
				{
					const int total_workload = static_cast<int>(entries_available_for_processing_count);
					const std::vector<float>::const_iterator output_buffer_it = output_buffer->begin();
					const std::vector<std::vector<float> >::iterator neuron_value_list_it = predicted_output_neuron_value_set->neuron_value_list.begin() + entries_copied_count;
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < total_workload; ++i)
					{
						std::vector<float>::const_iterator src_it = output_buffer_it + (i * output_neuron_count);
						std::vector<float>& value_list_dest = *(neuron_value_list_it + i);
						std::copy(src_it, src_it + output_neuron_count, value_list_dest.begin());
					}
				}

				entries_copied_count += entries_available_for_processing_count;
			}

			return predicted_output_neuron_value_set;
		}

		void network_tester_plain::actual_set_data(network_data_smart_ptr data)
		{
			net_data = data;
		}

		std::vector<layer_configuration_specific_snapshot_smart_ptr> network_tester_plain::actual_get_snapshot(std::vector<unsigned char>& input)
		{
			std::vector<layer_configuration_specific_snapshot_smart_ptr> res;

			const unsigned int input_neuron_count = layer_config_list[0].get_neuron_count();
			const unsigned int input_feature_map_count = static_cast<unsigned int>(current_scale_params->feature_map_count);
			const unsigned int neuron_count_per_input_feature_map = static_cast<unsigned int>(input.size() / current_scale_params->feature_map_count);

			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_buffers_pack;
			std::vector<additional_buffer_smart_ptr> output_buffer_list;
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_iterator layer_it = layer_list.begin();
				layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
				for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				{
					additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
						1,
						*layer_it,
						*input_config_it,
						*(input_config_it + 1),
						plain_config);
					input_buffer_and_additional_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
					output_buffer_list.push_back(output_buffer);
					++layer_it;
					++input_config_it;
				}
			}

			// Convert input
			{
				layer_configuration_specific_snapshot_smart_ptr input_elem(new layer_configuration_specific_snapshot(layer_config_list[0]));
				res.push_back(input_elem);

				const int elem_count = static_cast<int>(input_feature_map_count);
				const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
				const std::vector<float>::iterator input_elem_it_start = input_elem->data.begin();
				const std::vector<unsigned char>::const_iterator input_buf_it_start = input.begin();
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int feature_map_id = 0; feature_map_id < elem_count; ++feature_map_id)
				{
					std::vector<float>::iterator input_converted_buf_it = input_converted_buf_it_start + (feature_map_id * neuron_count_per_input_feature_map);
					std::vector<float>::iterator input_elem_it = input_elem_it_start + (feature_map_id * neuron_count_per_input_feature_map);
					std::vector<unsigned char>::const_iterator input_buf_it = input_buf_it_start + (feature_map_id * neuron_count_per_input_feature_map);

					float addition = current_scale_params->addition_list[feature_map_id];
					float multiplication = current_scale_params->multiplication_list[feature_map_id];
					for(unsigned int j = 0; j < neuron_count_per_input_feature_map; ++j)
					{
						float val = ((static_cast<float>(*input_buf_it) * (1.0F / 255.0F)) + addition) * multiplication;
						*input_converted_buf_it = val;
						*input_elem_it = val;

						input_buf_it++;
						input_converted_buf_it++;
						input_elem_it++;
					}
				}
			}

			// Run ann
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_iterator layer_it = layer_list.begin();
				layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
				std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_buffers_pack.begin();
				layer_data_list::const_iterator data_it = net_data->begin();
				std::vector<additional_buffer_smart_ptr>::iterator output_it = output_buffer_list.begin();
				for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				{
					(*it)->test(
						buffers_it->first,
						buffers_it->second,
						plain_config,
						*layer_it,
						*data_it,
						*input_config_it,
						*(input_config_it + 1),
						1);

					layer_configuration_specific_snapshot_smart_ptr new_elem(new layer_configuration_specific_snapshot(*(input_config_it + 1)));
					res.push_back(new_elem);

					std::copy((*output_it)->begin(), (*output_it)->end(), new_elem->data.begin());

					++layer_it;
					++input_config_it;
					++buffers_it;
					++data_it;
					++output_it;
				}
			}

			return res;
		}

		layer_configuration_specific_snapshot_smart_ptr network_tester_plain::actual_run(std::vector<unsigned char>& input)
		{
			layer_configuration_specific_snapshot_smart_ptr res(new layer_configuration_specific_snapshot(layer_config_list[layer_config_list.size() - 1]));

			const unsigned int input_neuron_count = layer_config_list[0].get_neuron_count();
			const unsigned int input_feature_map_count = current_scale_params->feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = static_cast<unsigned int>(input.size() / current_scale_params->feature_map_count);

			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_buffers_pack;
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_iterator layer_it = layer_list.begin();
				layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
				for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				{
					additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
						1,
						*layer_it,
						*input_config_it,
						*(input_config_it + 1),
						plain_config);
					input_buffer_and_additional_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
					++layer_it;
					++input_config_it;
				}
			}

			// Convert input
			{
				const int elem_count = static_cast<int>(input_feature_map_count);
				const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
				const std::vector<unsigned char>::const_iterator input_buf_it_start = input.begin();
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int feature_map_id = 0; feature_map_id < elem_count; ++feature_map_id)
				{
					std::vector<float>::iterator input_converted_buf_it = input_converted_buf_it_start + (feature_map_id * neuron_count_per_input_feature_map);
					std::vector<unsigned char>::const_iterator input_buf_it = input_buf_it_start + (feature_map_id * neuron_count_per_input_feature_map);

					float addition = current_scale_params->addition_list[feature_map_id];
					float multiplication = current_scale_params->multiplication_list[feature_map_id];
					for(unsigned int j = 0; j < neuron_count_per_input_feature_map; ++j)
					{
						*input_converted_buf_it = ((static_cast<float>(*input_buf_it) * (1.0F / 255.0F)) + addition) * multiplication;
						input_buf_it++;
						input_converted_buf_it++;
					}
				}
			}

			// Run ann
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_iterator layer_it = layer_list.begin();
				layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
				std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_buffers_pack.begin();
				layer_data_list::const_iterator data_it = net_data->begin();
				for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it, ++buffers_it, ++data_it)
				{
					(*it)->test(
						buffers_it->first,
						buffers_it->second,
						plain_config,
						*layer_it,
						*data_it,
						*input_config_it,
						*(input_config_it + 1),
						1);
				}
			}

			std::copy(output_buffer->begin(), output_buffer->end(), res->data.begin());

			return res;
		}

		void network_tester_plain::layer_config_list_modified()
		{
		}

		void network_tester_plain::update_buffers_configuration_testing(buffer_plain_size_configuration& buffer_configuration) const
		{
			for(std::vector<layer_data_smart_ptr>::const_iterator it = net_data->begin(); it != net_data->end(); ++it)
				for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
					buffer_configuration.add_constant_buffer(it2->size() * sizeof(float));

			const const_layer_list& layer_list = *schema;
			const_layer_list::const_iterator layer_it = layer_list.begin();
			layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
			for(std::vector<const_layer_tester_plain_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				(*it)->update_buffer_configuration(
					buffer_configuration,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config);
			}
		}
	}
}
