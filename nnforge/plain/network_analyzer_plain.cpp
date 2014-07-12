/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "network_analyzer_plain.h"

#include "layer_updater_plain_factory.h"
#include "../neural_network_exception.h"

#include <boost/format.hpp>
#include <cstring>

namespace nnforge
{
	namespace plain
	{
		network_analyzer_plain::network_analyzer_plain(
			network_schema_smart_ptr schema,
			plain_running_configuration_const_smart_ptr plain_config)
			: network_analyzer(schema)
			, plain_config(plain_config)
		{
			const const_layer_list& layer_list = *schema;
			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				updater_list.push_back(plain::single_layer_updater_plain_factory::get_const_instance().get_updater_plain_layer((*it)->get_uuid()));
		}

		network_analyzer_plain::~network_analyzer_plain()
		{
		}

		void network_analyzer_plain::layer_config_list_modified()
		{
			input_buffer_and_additional_updater_buffers_pack.clear();
			output_errors_buffers.clear();

			const unsigned int input_neuron_count = layer_config_list.front().get_neuron_count();
			const unsigned int output_neuron_count = layer_config_list.back().get_neuron_count();
			input_converted_buf = additional_buffer_smart_ptr(new std::vector<float>(input_neuron_count));
			initial_error_buf = additional_buffer_smart_ptr(new std::vector<float>(output_neuron_count));

			additional_buffer_smart_ptr output_buffer = input_converted_buf;

			const const_layer_list& layer_list = *schema;
			const_layer_list::const_iterator layer_it = layer_list.begin();
			layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
			for(const_layer_updater_plain_list::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it)
			{
				updater_additional_buffer_set additional_buffers = (*it)->allocate_additional_buffers(
					1,
					*layer_it,
					*input_config_it,
					*(input_config_it + 1),
					plain_config,
					true);
				input_buffer_and_additional_updater_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = additional_buffers.output_neurons_buffer;
			}

			{
				additional_buffer_smart_ptr output_errors = initial_error_buf;
				for(std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::reverse_iterator it = input_buffer_and_additional_updater_buffers_pack.rbegin(); it != input_buffer_and_additional_updater_buffers_pack.rend(); ++it)
				{
					output_errors_buffers.push_back(output_errors);
					if (it->second.input_errors_buffer != 0)
						output_errors = it->second.input_errors_buffer;
					else
						it->second.input_errors_buffer = output_errors;
				}
			}
		}

		void network_analyzer_plain::actual_set_data(network_data_smart_ptr data)
		{
			this->data = data;
		}

		void network_analyzer_plain::actual_set_input_data(
			const void * input,
			neuron_data_type::input_type type_code)
		{
			const unsigned int input_neuron_count = layer_config_list[0].get_neuron_count();

			const int elem_count = static_cast<int>(input_neuron_count);
			const std::vector<float>::iterator input_converted_buf_it_start = input_converted_buf->begin();
			if (type_code == neuron_data_type::type_byte)
			{
				const unsigned char * const input_buf_it_start = static_cast<const unsigned char *>(input);
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < elem_count; ++i)
					*(input_converted_buf_it_start + i) = static_cast<float>(*(input_buf_it_start + i)) * (1.0F / 255.0F);
			}
			else if (type_code == neuron_data_type::type_float)
			{
				const float * const input_buf_it_start = static_cast<const float *>(input);
				#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
				for(int i = 0; i < elem_count; ++i)
					*(input_converted_buf_it_start + i) = *(input_buf_it_start + i);
			}
			else
				throw neural_network_exception((boost::format("actual_set_input_data cannot handle input neurons of type %1%") % type_code).str());

			const const_layer_list& layer_list = *schema;
			const_layer_list::const_iterator layer_it = layer_list.begin();
			layer_configuration_specific_list::const_iterator input_config_it = layer_config_list.begin();
			std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.begin();
			layer_data_list::const_iterator data_it = data->begin();
			for(std::vector<const_layer_updater_plain_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++layer_it, ++input_config_it, ++updater_buffers_it, ++data_it)
			{
				(*it)->test(
					updater_buffers_it->first,
					updater_buffers_it->second.output_neurons_buffer,
					updater_buffers_it->second.additional_buffers,
					plain_config,
					*layer_it,
					*data_it,
					*input_config_it,
					*(input_config_it + 1),
					1,
					0);
			}
		}

		std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> network_analyzer_plain::actual_run_backprop(
			const layer_configuration_specific_snapshot& output_data,
			const std::vector<unsigned int>& output_offset_list,
			unsigned int output_layer_id,
			const std::vector<std::pair<unsigned int, unsigned int> >& input_rectangle_borders)
		{
			std::vector<additional_buffer_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin() + (output_errors_buffers.size() - output_layer_id - 1);
			// Initialize output errors
			{
				std::fill((*output_errors_it)->begin(), (*output_errors_it)->end(), 0.0F);
				float * dst = &(*(*output_errors_it)->begin());
				const layer_configuration_specific& output_config = layer_config_list[output_layer_id + 1];
				int sequential_chunk_dimension_count = -1;
				unsigned int sequential_copy_elem_count = 1;
				while (sequential_chunk_dimension_count < (int)output_offset_list.size() - 1)
				{
					++sequential_chunk_dimension_count;
					sequential_copy_elem_count *= output_data.config.dimension_sizes[sequential_chunk_dimension_count];
					if (output_data.config.dimension_sizes[sequential_chunk_dimension_count] != output_config.dimension_sizes[sequential_chunk_dimension_count])
						break;
				}
				++sequential_chunk_dimension_count;

				std::vector<float>::const_iterator src_it = output_data.data.begin();
				for(unsigned int feature_map_id = 0; feature_map_id < output_data.config.feature_map_count; ++feature_map_id)
				{
					unsigned int dst_fm_offset = feature_map_id * output_config.get_neuron_count_per_feature_map();
					std::vector<unsigned int> src_list(output_offset_list.size(), 0);

					bool cont = true;
					while (cont)
					{
						bool should_copy = false;
						for(std::vector<float>::const_iterator it = src_it; it != src_it + sequential_copy_elem_count; ++it)
						{
							if (*src_it != 0.0F)
							{
								should_copy = true;
								break;
							}
						}
						if (should_copy)
						{
							std::vector<unsigned int> dst_offset_list(output_offset_list);
							for(unsigned int i = sequential_chunk_dimension_count; i < dst_offset_list.size(); ++i)
								dst_offset_list[i] += src_list[i];
							memcpy(dst + dst_fm_offset + output_config.get_pos(dst_offset_list), &(*src_it), sequential_copy_elem_count * sizeof(float));
						};

						cont = false;
						for(int i = sequential_chunk_dimension_count; i < src_list.size(); ++i)
						{
							src_list[i]++;
							if (src_list[i] < output_data.config.dimension_sizes[i])
							{
								cont = true;
								break;
							}
							else
								src_list[i] = 0;
						}

						src_it += sequential_copy_elem_count;
					}
				}
			}


			// Run backward and update weights
			{
				const const_layer_list& layer_list = *schema;
				const_layer_list::const_reverse_iterator layer_it = layer_list.rbegin() + (layer_list.size() - output_layer_id - 1);
				std::vector<std::pair<additional_buffer_smart_ptr, updater_additional_buffer_set> >::reverse_iterator updater_buffers_it = input_buffer_and_additional_updater_buffers_pack.rbegin() + (input_buffer_and_additional_updater_buffers_pack.size() - output_layer_id - 1);
				layer_configuration_specific_list::const_reverse_iterator input_config_it = layer_config_list.rbegin() + (layer_config_list.size() - output_layer_id - 2);
				layer_data_list::reverse_iterator data_it = data->rbegin() + (data->size() - output_layer_id - 1);
				for(std::vector<const_layer_updater_plain_smart_ptr>::const_reverse_iterator it = updater_list.rbegin() + (updater_list.size() - output_layer_id - 1); it != updater_list.rend(); ++it, ++layer_it, ++input_config_it, ++updater_buffers_it, ++data_it, ++output_errors_it)
				{
					(*it)->backprop(
						updater_buffers_it->second.input_errors_buffer,
						updater_buffers_it->first,
						*output_errors_it,
						updater_buffers_it->second.output_neurons_buffer,
						updater_buffers_it->second.additional_buffers,
						plain_config,
						*layer_it,
						*data_it,
						*(input_config_it + 1),
						*input_config_it,
						1);
				}
			}

			layer_configuration_specific_snapshot_smart_ptr res(new layer_configuration_specific_snapshot());
			layer_configuration_specific_snapshot_smart_ptr input_data(new layer_configuration_specific_snapshot());

			// Copy input errors
			{
				res->config.feature_map_count = layer_config_list.front().feature_map_count;
				input_data->config.feature_map_count = layer_config_list.front().feature_map_count;
				unsigned int elem_count = res->config.feature_map_count;
				for(int i = 0; i < input_rectangle_borders.size(); ++i)
				{
					unsigned int val = input_rectangle_borders[i].second - input_rectangle_borders[i].first;
					elem_count *= val;
					res->config.dimension_sizes.push_back(val);
					input_data->config.dimension_sizes.push_back(val);
				}
				res->data.resize(elem_count);
				input_data->data.resize(elem_count);

				additional_buffer_smart_ptr input_errors_buf = input_buffer_and_additional_updater_buffers_pack.front().second.input_errors_buffer;
				float * src = &(*input_errors_buf->begin());
				float * src_input_data = &(*input_converted_buf->begin());
				const layer_configuration_specific& input_config = layer_config_list.front();
				int sequential_chunk_dimension_count = -1;
				unsigned int sequential_copy_elem_count = 1;
				while (sequential_chunk_dimension_count < (int)input_config.dimension_sizes.size() - 1)
				{
					++sequential_chunk_dimension_count;
					sequential_copy_elem_count *= res->config.dimension_sizes[sequential_chunk_dimension_count];
					if (res->config.dimension_sizes[sequential_chunk_dimension_count] != input_config.dimension_sizes[sequential_chunk_dimension_count])
						break;
				}
				++sequential_chunk_dimension_count;

				std::vector<float>::iterator dst_it = res->data.begin();
				std::vector<float>::iterator dst_input_data_it = input_data->data.begin();
				for(unsigned int feature_map_id = 0; feature_map_id < input_config.feature_map_count; ++feature_map_id)
				{
					unsigned int src_fm_offset = feature_map_id * input_config.get_neuron_count_per_feature_map();
					std::vector<unsigned int> dst_list(input_rectangle_borders.size(), 0);

					bool cont = true;
					while (cont)
					{
						std::vector<unsigned int> src_offset_list(input_rectangle_borders.size());
						for(int i = 0; i < src_offset_list.size(); ++i)
							src_offset_list[i] = input_rectangle_borders[i].first;
						for(unsigned int i = sequential_chunk_dimension_count; i < src_offset_list.size(); ++i)
							src_offset_list[i] += dst_list[i];
						memcpy(&(*dst_it), src + src_fm_offset + input_config.get_pos(src_offset_list), sequential_copy_elem_count * sizeof(float));
						memcpy(&(*dst_input_data_it), src_input_data + src_fm_offset + input_config.get_pos(src_offset_list), sequential_copy_elem_count * sizeof(float));

						cont = false;
						for(int i = sequential_chunk_dimension_count; i < dst_list.size(); ++i)
						{
							dst_list[i]++;
							if (dst_list[i] < res->config.dimension_sizes[i])
							{
								cont = true;
								break;
							}
							else
								dst_list[i] = 0;
						}

						dst_it += sequential_copy_elem_count;
						dst_input_data_it += sequential_copy_elem_count;
					}
				}
			}

			return std::make_pair(res, input_data);
		}
	}
}
