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

#include "hessian_calculator_plain.h"

#include <boost/format.hpp>

#include "layer_tester_plain_factory.h"
#include "layer_hessian_plain_factory.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		hessian_calculator_plain::hessian_calculator_plain(
			network_schema_smart_ptr schema,
			plain_running_configuration_const_smart_ptr plain_config)
			: hessian_calculator(schema)
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
				hessian_list.push_back(single_layer_hessian_plain_factory::get_const_instance().get_hessian_plain_layer((*it)->get_uuid()));
		}

		hessian_calculator_plain::~hessian_calculator_plain()
		{
		}

		network_data_smart_ptr hessian_calculator_plain::actual_get_hessian(
			supervised_data_reader& reader,
			network_data_smart_ptr data,
			unsigned int hessian_entry_to_process_count)
		{
			network_data_smart_ptr hessian(new network_data(*schema));

			reader.reset();

			const unsigned int input_neuron_count = reader.get_input_configuration().get_neuron_count();
			const unsigned int output_neuron_count = reader.get_output_configuration().get_neuron_count();
			const unsigned int input_feature_map_count = reader.get_input_configuration().feature_map_count;
			const unsigned int neuron_count_per_input_feature_map = reader.get_input_configuration().get_neuron_count_per_feature_map();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			buffer_plain_size_configuration buffers_config;
			update_buffers_configuration(buffers_config);
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // initial error
			for(std::vector<layer_data_smart_ptr>::const_iterator it = data->begin(); it != data->end(); ++it)
			{
				for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
				{
					buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // data
					buffers_config.add_constant_buffer(it2->size() * sizeof(float)); // hessian
				}
			}

			unsigned int max_entry_count = std::min<unsigned int>(plain_config->get_max_entry_count(buffers_config), hessian_entry_to_process_count);

			std::vector<unsigned char> input_buf(max_entry_count * input_neuron_count * input_neuron_elem_size);
			additional_buffer_smart_ptr initial_error_buf(new std::vector<float>(max_entry_count * output_neuron_count));
			additional_buffer_smart_ptr input_converted_buf(new std::vector<float>(input_neuron_count * max_entry_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> > input_buffer_and_additional_testing_buffers_pack;
			std::vector<std::pair<additional_buffer_smart_ptr, hessian_additional_buffer_set> > input_buffer_and_additional_hessian_buffers_pack;
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
					input_buffer_and_additional_testing_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				}
				for(const_layer_hessian_plain_list::const_iterator it = hessian_list.begin(); it != hessian_list.end(); ++it, ++layer_it, ++input_config_it)
				{
					hessian_additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
						max_entry_count,
						*layer_it,
						*input_config_it,
						*(input_config_it + 1),
						plain_config,
						(it != hessian_list.begin()));
					input_buffer_and_additional_hessian_buffers_pack.push_back(std::make_pair<additional_buffer_smart_ptr, hessian_additional_buffer_set>(output_buffer, additional_buffers));
					output_buffer = additional_buffers.output_neurons_buffer;
				}
			}
			{
				additional_buffer_smart_ptr output_errors = initial_error_buf;
				for(std::vector<std::pair<additional_buffer_smart_ptr, hessian_additional_buffer_set> >::reverse_iterator it = input_buffer_and_additional_hessian_buffers_pack.rbegin(); it != input_buffer_and_additional_hessian_buffers_pack.rend() - 1; ++it)
				{
					if (it->second.input_errors_buffer != 0)
						output_errors = it->second.input_errors_buffer;
					else
						it->second.input_errors_buffer = output_errors;
				}
			}

			bool entries_remained_for_loading = true;
			unsigned int entries_read_count = 0;
			while (entries_remained_for_loading && (entries_read_count < hessian_entry_to_process_count))
			{
				unsigned int entries_available_for_processing_count = 0;
				while((entries_available_for_processing_count < max_entry_count) && (entries_read_count < hessian_entry_to_process_count))
				{
					bool entry_read = reader.read(
						&(*(input_buf.begin() + (input_neuron_count * entries_available_for_processing_count * input_neuron_elem_size))),
						0);

					if (!entry_read)
						throw neural_network_exception((boost::format("Unable to read %1% entries to calculate hessian, %2% read") % hessian_entry_to_process_count % entries_read_count).str());

					entries_available_for_processing_count++;
					entries_read_count++;
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
						throw neural_network_exception((boost::format("actual_get_hessian cannot handle input neurons of type %1%") % type_code).str());
				}

				// Run ann
				{
					const const_layer_list& layer_list = *schema;
					const_layer_list::const_iterator layer_it = layer_list.begin();
					layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
					std::vector<std::pair<additional_buffer_smart_ptr, additional_buffer_set> >::iterator buffers_it = input_buffer_and_additional_testing_buffers_pack.begin();
					layer_data_list::const_iterator data_it = data->begin();
					// Run testing
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
					// Forward hessian
					std::vector<std::pair<additional_buffer_smart_ptr, hessian_additional_buffer_set> >::iterator hessian_buffers_it = input_buffer_and_additional_hessian_buffers_pack.begin();
					for(std::vector<const_layer_hessian_plain_smart_ptr>::const_iterator it = hessian_list.begin(); it != hessian_list.end(); ++it, ++layer_it, ++input_config_it, ++hessian_buffers_it, ++data_it)
					{
						(*it)->test(
							hessian_buffers_it->first,
							hessian_buffers_it->second.output_neurons_buffer,
							hessian_buffers_it->second.additional_buffers,
							plain_config,
							*layer_it,
							*data_it,
							*input_config_it,
							*(input_config_it + 1),
							entries_available_for_processing_count);
					}
				}

				// Set initial errors to 1.0F
				{
					const int elem_count = static_cast<int>(entries_available_for_processing_count * output_neuron_count);
					const std::vector<float>::iterator initial_error_buf_it = initial_error_buf->begin();
					#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
					for(int i = 0; i < elem_count; ++i)
					{
						*(initial_error_buf_it + i) = 1.0F;
					}
				}

				// Backward hessian
				{
					const const_layer_list& layer_list = *schema;
					const_layer_list::const_reverse_iterator layer_it = layer_list.rbegin();
					std::vector<std::pair<additional_buffer_smart_ptr, hessian_additional_buffer_set> >::reverse_iterator hessian_buffers_it = input_buffer_and_additional_hessian_buffers_pack.rbegin();
					layer_configuration_specific_list::const_reverse_iterator input_config_it = layer_config_list.rbegin();
					layer_data_list::const_reverse_iterator data_it = data->rbegin();
					layer_data_list::reverse_iterator hessian_data_it = hessian->rbegin();
					additional_buffer_smart_ptr output_errors = initial_error_buf;
					for(std::vector<const_layer_hessian_plain_smart_ptr>::const_reverse_iterator it = hessian_list.rbegin(); it != hessian_list.rend(); ++it, ++layer_it, ++input_config_it, ++hessian_buffers_it, ++data_it, ++hessian_data_it)
					{
						if (it != hessian_list.rend() - 1)
						{
							(*it)->backprop(
								hessian_buffers_it->second.input_errors_buffer,
								output_errors,
								hessian_buffers_it->second.output_neurons_buffer,
								hessian_buffers_it->second.additional_buffers,
								plain_config,
								*layer_it,
								*data_it,
								*(input_config_it + 1),
								*input_config_it,
								entries_available_for_processing_count);
						}

						(*it)->update_hessian(
							hessian_buffers_it->first,
							output_errors,
							hessian_buffers_it->second.additional_buffers,
							*hessian_data_it,
							plain_config,
							*layer_it,
							*(input_config_it + 1),
							*input_config_it,
							entries_available_for_processing_count);

						output_errors = hessian_buffers_it->second.input_errors_buffer;
					}
				}
			}

			const float mult = 1.0F / static_cast<float>(entries_read_count);
			{
				for(layer_data_list::iterator it = hessian->begin(); it != hessian->end(); ++it)
				{
					for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
					{
						const std::vector<float>::iterator it3 = it2->begin();
						const int elem_count = static_cast<int>(it2->size());
						#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
						for(int i = 0; i < elem_count; ++i)
						{
							*(it3 + i) *= mult;
						}
					}
				}
			}

			return hessian;
		}

		void hessian_calculator_plain::layer_config_list_modified()
		{
		}

		void hessian_calculator_plain::update_buffers_configuration(buffer_plain_size_configuration& buffer_configuration) const
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
			for(const_layer_hessian_plain_list::const_iterator it = hessian_list.begin(); it != hessian_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				(*it)->update_buffer_configuration(
					buffer_configuration,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config,
					(it != hessian_list.begin()));
			}
		}
	}
}
