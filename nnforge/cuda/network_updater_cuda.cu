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

#include "network_updater_cuda.h"

#include "neural_network_cuda_exception.h"
#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "util_cuda.h"
#include "cuda_event.h"
#include "layer_updater_schema_factory.h"
#include "weight_vector_bound_cuda_factory.h"
#include "supervised_data_reader_async_helper.h"
#include "error_function_updater_cuda_factory.h"

#include "../nn_types.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>
#include <stack>

#include "../debug_util.h"
#include <boost/filesystem.hpp>

namespace nnforge
{
	namespace cuda
	{
		__global__ void convert_compacted_to_raw_upd_kernel(
			const uchar4 * __restrict input,
			float4 * __restrict output,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				uchar4 inp = input[elem_id];
				float4 val;
				val.x = inp.x * (1.0F / 255.0F);
				val.y = inp.y * (1.0F / 255.0F);
				val.z = inp.z * (1.0F / 255.0F);
				val.w = inp.w * (1.0F / 255.0F);
				output[elem_id] = val;
			}
		}

		__global__ void dropout_kernel(
			float * __restrict neurons,
			const float * __restrict random_buf,
			float dropout_rate,
			int offset,
			unsigned int mask,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				unsigned int random_elem_id = (elem_id + offset) & mask;
				if (random_buf[random_elem_id] < dropout_rate)
					neurons[elem_id] = 0.0F;
			}
		}

		unsigned int network_updater_cuda::max_entry_count_in_single_batch = 1024;

		network_updater_cuda::network_updater_cuda(
			network_schema_smart_ptr schema,
			const_error_function_smart_ptr ef,
			const std::map<unsigned int, float>& layer_to_dropout_rate_map,
			const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
			float weight_decay,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_updater(schema, ef, layer_to_dropout_rate_map, layer_to_weight_vector_bound_map, weight_decay)
			, cuda_config(cuda_config)
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
				testing_schemas.push_back(single_layer_testing_schema_factory::get_const_instance().create_testing_schema_layer(*it, cuda_config));

			for(const_layer_list::const_iterator it = start_layer_nonempty_weights_iterator; it != layer_list.end(); ++it)
				updater_schemas.push_back(single_layer_updater_schema_factory::get_const_instance().create_updater_schema_layer(*it, cuda_config));

			for(std::map<unsigned int, weight_vector_bound>::const_iterator it = this->layer_to_weight_vector_bound_map.begin(); it != this->layer_to_weight_vector_bound_map.end(); ++it)
			{
				unsigned int layer_id = it->first;
				if (layer_id < testing_layer_count)
					throw neural_network_exception((boost::format("Weight vector bound is specified for layer %1% while it is in testing part (consisting of %2% layers) of the updater") % layer_id  % testing_layer_count).str());

				weight_vector_bounds.insert(std::make_pair(layer_id, single_weight_vector_bound_factory::get_const_instance().create_weight_vector_bound(layer_list[layer_id], cuda_config)));
			}

			ef_updater = single_error_function_updater_cuda_factory::get_const_instance().get_error_function_updater_cuda(ef->get_uuid());

			setup_network_cuda();

			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
				testing_schema_data.push_back((*it)->get_schema_buffers());

			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it)
				updater_schema_data.push_back((*it)->get_schema_buffers());
		}

		network_updater_cuda::~network_updater_cuda()
		{
		}

		void network_updater_cuda::setup_network_cuda()
		{
			command_stream = cuda_stream_smart_ptr(new cuda_stream());
			data_stream = cuda_stream_smart_ptr(new cuda_stream());
		}

		std::vector<testing_result_smart_ptr> network_updater_cuda::actual_update(
			supervised_data_reader& reader,
			const std::vector<network_data_smart_ptr>& learning_rate_vector_list,
			std::vector<network_data_smart_ptr>& data_list)
		{
			std::vector<testing_result_smart_ptr> res;

			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();
			layer_configuration_specific output_configuration = reader.get_output_configuration();

			unsigned int input_neuron_count = input_configuration.get_neuron_count();
			unsigned int output_neuron_count = output_configuration.get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			unsigned int updater_entry_count = static_cast<unsigned int>(data_list.size());
			if (updater_entry_count == 0)
				return res;

			for(unsigned int i = 0; i < learning_rate_vector_list.size(); ++i)
				res.push_back(testing_result_smart_ptr(new testing_result(ef)));

			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > net_data = get_data(data_list);
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > learning_rate_data = get_learning_rate(learning_rate_vector_list);

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration(buffers_config, updater_entry_count);

			buffers_config.add_per_entry_linear_addressing_through_texture(layer_config_list[testing_layer_count].get_neuron_count()); // This is for the first updater to safely read input data through the texture
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // initial error
			buffers_config.add_constant_buffer(sizeof(double) * updater_entry_count); // error buffer
			if (!random_uniform_list.empty())
				buffers_config.add_constant_buffer(random_uniform_list.size() * sizeof(float)); // random_uniform_list

			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = learning_rate_data.begin(); it != learning_rate_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			unsigned int max_entry_count = std::min<unsigned int>(std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), reader.get_entry_count()), max_entry_count_in_single_batch);

			cuda_linear_buffer_device_smart_ptr input_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * input_neuron_elem_size)),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * input_neuron_elem_size)),
			};

			cuda_linear_buffer_device_smart_ptr output_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_count * sizeof(float))),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_count * sizeof(float))),
			};

			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr initial_error_buf(new cuda_linear_buffer_device(output_neuron_count * updater_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr error_buf(new cuda_linear_buffer_device(updater_entry_count * sizeof(double)));

			cuda_linear_buffer_device_smart_ptr random_uniform_buf;
			if (!random_uniform_list.empty())
			{
				random_uniform_buf = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(random_uniform_list.size() * sizeof(float)));
				cuda_safe_call(cudaMemcpyAsync(*random_uniform_buf, &(*random_uniform_list.begin()), random_uniform_list.size() * sizeof(float), cudaMemcpyHostToDevice, *command_stream));
			}

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > testing_input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_count);
				testing_input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> > updater_input_and_all_buffers_pack;
			for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it)
			{
				layer_updater_cuda::buffer_set all_buffers = (*it)->allocate_all_buffers(updater_entry_count);
				updater_input_and_all_buffers_pack.push_back(std::make_pair(output_buffer, all_buffers));
				output_buffer = all_buffers.output_neurons_buffer;
			}

			std::vector<cuda_linear_buffer_device_smart_ptr> output_errors_buffers;
			cuda_linear_buffer_device_smart_ptr output_errors = initial_error_buf;
			for(std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator it = updater_input_and_all_buffers_pack.rbegin(); it != updater_input_and_all_buffers_pack.rend(); ++it)
			{
				output_errors_buffers.push_back(output_errors);
				layer_updater_cuda::buffer_set& all_buffers = it->second;

				if (all_buffers.input_errors_buffer != 0)
					output_errors = all_buffers.input_errors_buffer;
			}

			std::map<unsigned int, std::vector<cuda_linear_buffer_device_smart_ptr> > weight_vector_bound_buffers;
			for(std::map<unsigned int, weight_vector_bound_cuda_smart_ptr>::const_iterator it = weight_vector_bounds.begin(); it != weight_vector_bounds.end(); ++it)
				weight_vector_bound_buffers.insert(std::make_pair(it->first, it->second->allocate_additional_buffers(max_entry_count)));

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * input_neuron_elem_size));
			unsigned char * input = *input_host_buf;
			cuda_linear_buffer_host_smart_ptr output_host_buf(new cuda_linear_buffer_host(output_neuron_count * max_entry_count * sizeof(float)));
			float * output = *output_host_buf;

			// zero mse
			cuda_util::set_with_value(
				*cuda_config,
				(double *)(*error_buf),
				0.0,
				updater_entry_count,
				*command_stream);

			unsigned int current_data_slot = 0;
			unsigned int current_command_slot = 1;
			unsigned int entries_available_for_copy_in_count = reader.get_entry_count();
			unsigned int entries_available_for_processing_count = 0;
			cuda_event data_processed_event;
			cuda_event input_copied_event;
			if (cuda_config->is_flush_required())
			{
				cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
				cuda_safe_call(cudaEventQuery(data_processed_event));
			}

			random_generator gen = rnd::get_random_generator();
			nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(random_uniform_list.size() - 1));
			unsigned int mask = static_cast<unsigned int>(random_uniform_list.size() - 1);
			unsigned int entries_processed_count = 0;
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0))
			{
				supervised_data_reader_async_helper async_reader;
				if (entries_available_for_copy_in_count > 0)
				{
					unsigned int entries_to_read_count = std::min<unsigned int>(max_entry_count, entries_available_for_copy_in_count);
					async_reader.fun = supervised_data_reader_functor(
						entries_to_read_count,
						&reader,
						input,
						output,
						*(input_buf[current_data_slot]),
						*(output_buf[current_data_slot]),
						*data_stream);
					async_reader.start();
				}

				if (entries_available_for_processing_count > 0)
				{
					// Convert input
					if (type_code == neuron_data_type::type_byte)
					{
						int elem_count = (input_neuron_count * entries_available_for_processing_count + 3) / 4;
						std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							elem_count);
						convert_compacted_to_raw_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
							*input_buf[current_command_slot],
							*input_converted_buf,
							elem_count);
					}
					else if (type_code == neuron_data_type::type_float)
					{
						cuda_safe_call(cudaMemcpyAsync(
							*input_converted_buf,
							*input_buf[current_command_slot],
							input_neuron_count * entries_available_for_processing_count * sizeof(float),
							cudaMemcpyDeviceToDevice,
							*command_stream));
					}
					else throw neural_network_exception((boost::format("actual_update cannot handle input neurons of type %1%") % type_code).str());

					// Run ann
					{
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = testing_input_and_additional_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = testing_schema_data.begin();
						unsigned int layer_id = 0;
						layer_configuration_specific_list::const_iterator layer_config_it = layer_config_list.begin();
						for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++schema_data_it, ++layer_id, ++layer_config_it)
						{
							std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
							if (dropout_it != layer_to_dropout_rate_map.end())
							{
								unsigned int offset = dist(gen);
								enqueue_dropout(
									*command_stream,
									random_uniform_buf,
									input_and_additional_buffers_pack_it->first,
									dropout_it->second,
									mask,
									entries_available_for_processing_count * layer_config_it->get_neuron_count(),
									offset);
							}

							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								std::vector<const_cuda_linear_buffer_device_smart_ptr>(),
								input_and_additional_buffers_pack_it->first,
								input_and_additional_buffers_pack_it->second,
								entries_available_for_processing_count);
						}
					}

					// Apply dropout to the input of the first updater layer
					{
						std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(testing_layer_count);
						if (dropout_it != layer_to_dropout_rate_map.end())
						{
							unsigned int offset = dist(gen);
							enqueue_dropout(
								*command_stream,
								random_uniform_buf,
								updater_input_and_all_buffers_pack[0].first,
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
							std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.begin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = updater_schema_data.begin();
							unsigned int layer_id = testing_layer_count;
							layer_configuration_specific_list::const_iterator layer_config_it = layer_config_list.begin() + testing_layer_count;
							for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++net_data_it, ++layer_id, ++layer_config_it)
							{
								if (it != updater_list.begin())
								{
									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = dist(gen);
										offset_list.push(offset);
										enqueue_dropout(
											*command_stream,
											random_uniform_buf,
											input_and_all_buffers_pack_it->first,
											dropout_it->second,
											mask,
											updater_entry_count * layer_config_it->get_neuron_count(),
											offset);
									}
								}

								(*it)->enqueue_test(
									it == updater_list.begin() ? input_entry_id : 0,
									*command_stream,
									*schema_data_it,
									*net_data_it,
									input_and_all_buffers_pack_it->first,
									input_and_all_buffers_pack_it->second.output_neurons_buffer,
									input_and_all_buffers_pack_it->second.additional_buffers,
									input_and_all_buffers_pack_it->second.dynamic_memobjects,
									updater_entry_count);
							}
						}

						// Compute errors
						{
							ef_updater->enqueue_update_error_and_gradient(
								*command_stream,
								initial_error_buf,
								error_buf,
								output_buf[current_command_slot],
								output_buffer,
								input_entry_id,
								output_neuron_count,
								updater_entry_count);
						}

						// Backward updater
						{
							std::vector<cuda_linear_buffer_device_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin();
							std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.rbegin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_it = net_data.rbegin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator learning_rate_data_it = learning_rate_data.rbegin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator schema_data_it = updater_schema_data.rbegin();
							unsigned int reverse_layer_id = static_cast<unsigned int>(updater_list.size() + testing_layer_count) - 1;
							layer_configuration_specific_list::const_reverse_iterator layer_config_it = layer_config_list.rbegin() + 1;
							std::vector<std::vector<unsigned int> >::reverse_iterator incoming_weight_count_it = incoming_weight_count_per_output_neuron_list_list.rbegin();
							for(std::vector<layer_updater_cuda_smart_ptr>::reverse_iterator it = updater_list.rbegin(); it != updater_list.rend(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++learning_rate_data_it, ++output_errors_it, ++net_data_it, --reverse_layer_id, ++layer_config_it, ++incoming_weight_count_it)
							{
								if (it != (updater_list.rend() - 1))
								{
									(*it)->enqueue_backprop(
										*command_stream,
										*schema_data_it,
										*net_data_it,
										input_and_all_buffers_pack_it->second.output_neurons_buffer,
										input_and_all_buffers_pack_it->first,
										*output_errors_it,
										input_and_all_buffers_pack_it->second.input_errors_buffer,
										input_and_all_buffers_pack_it->second.additional_buffers,
										input_and_all_buffers_pack_it->second.dynamic_memobjects,
										updater_entry_count);

									/*
									{
										cuda_linear_buffer_device_smart_ptr buf = (input_and_all_buffers_pack_it->second.input_errors_buffer == 0) ? *output_errors_it : input_and_all_buffers_pack_it->second.input_errors_buffer;
										std::vector<float> inp_err(buf->get_size() / sizeof(float));
										cuda_safe_call(cudaMemcpyAsync(&(*inp_err.begin()), *buf, inp_err.size() * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));
										cuda_safe_call(cudaStreamSynchronize(*command_stream));
										
										boost::filesystem::path dir = "Debug";
										dir /= "GPU";
										boost::filesystem::create_directories(dir);
										debug_util::dump_list(
											&(*inp_err.begin()),
											inp_err.size(),
											(dir / (boost::format("input_errors_%1%.txt") % reverse_layer_id).str()).string().c_str());
									}
									*/

									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(reverse_layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = offset_list.top();
										offset_list.pop();
										enqueue_dropout(
											*command_stream,
											random_uniform_buf,
											(input_and_all_buffers_pack_it->second.input_errors_buffer == 0) ? *output_errors_it : input_and_all_buffers_pack_it->second.input_errors_buffer,
											dropout_it->second,
											mask,
											updater_entry_count * layer_config_it->get_neuron_count(),
											offset);
									}
								}

								(*it)->enqueue_update_weights(
									(it == (updater_list.rend() - 1)) ? input_entry_id : 0,
									*command_stream,
									*net_data_it,
									*schema_data_it,
									*learning_rate_data_it,
									*output_errors_it,
									input_and_all_buffers_pack_it->first,
									input_and_all_buffers_pack_it->second.additional_buffers,
									input_and_all_buffers_pack_it->second.dynamic_memobjects,
									updater_entry_count,
									weight_decay);

								weight_vector_bound_map::iterator bound_it = weight_vector_bounds.find(reverse_layer_id);
								if (bound_it != weight_vector_bounds.end())
								{
									const weight_vector_bound& bound = layer_to_weight_vector_bound_map.find(reverse_layer_id)->second;
									const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers = weight_vector_bound_buffers.find(reverse_layer_id)->second;
									bound_it->second->enqueue_normalize_weights(
										*command_stream,
										bound,
										*net_data_it,
										additional_buffers,
										updater_entry_count,
										*incoming_weight_count_it);
								}
							}
						}

						if (((input_entry_id % 16) == 1) && cuda_config->is_flush_required())
						{
							cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
							cuda_safe_call(cudaEventQuery(data_processed_event));
						}
					} // for(unsigned int input_entry_id

					entries_processed_count += entries_available_for_processing_count;

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cuda_safe_call(cudaEventQuery(data_processed_event));
					}
				} // if (entries_available_for_processing_count > 0)

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
					entries_read_count = async_reader.wait();

				cuda_safe_call(cudaStreamSynchronize(*data_stream));
				cuda_safe_call(cudaStreamSynchronize(*command_stream));

				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;
			}

			read_data(net_data, data_list, *command_stream);

			std::vector<double> error_list(updater_entry_count);
			cuda_safe_call(cudaMemcpyAsync(&(*error_list.begin()), *error_buf, error_list.size() * sizeof(double), cudaMemcpyDeviceToHost, *command_stream));
			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			for(unsigned int i = 0; i < updater_entry_count; ++i)
				res[i]->init(error_list[i], entries_processed_count);

			return res;
		}

		void network_updater_cuda::layer_config_list_modified()
		{
			layer_configuration_specific_list::const_iterator it_conf = layer_config_list.begin();

			tester_list.clear();
			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it, ++it_conf)
			{
				tester_list.push_back(
					(*it)->create_tester(
						*it_conf,
						*(it_conf + 1)));
			}

			updater_list.clear();
			incoming_weight_count_per_output_neuron_list_list.clear();
			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it, ++it_conf)
			{
				updater_list.push_back(
					(*it)->create_updater(
						*it_conf,
						*(it_conf + 1),
						(it_conf > layer_config_list.begin() + testing_layer_count),
						(it_conf > layer_config_list.begin() + testing_layer_count)));
				incoming_weight_count_per_output_neuron_list_list.push_back(updater_list.back()->get_incoming_weight_count_per_output_neuron_list());
			}
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::get_data(const std::vector<network_data_smart_ptr>& data) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			for(int i = 0; i < updater_list.size(); ++i)
			{
				std::vector<layer_data_smart_ptr> data_list; 
				for(int j = 0; j < data.size(); ++j)
					data_list.push_back(data[j]->at(i + testing_layer_count));
				std::vector<cuda_linear_buffer_device_smart_ptr> device_data = updater_list[i]->get_data(data_list);
				res.push_back(device_data);
			}

			return res;
		}

		std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::get_learning_rate(const std::vector<network_data_smart_ptr>& learning_rate) const
		{
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > res;

			for(int i = 0; i < updater_list.size(); ++i)
			{
				std::vector<const_layer_data_smart_ptr> data_list; 
				for(int j = 0; j < learning_rate.size(); ++j)
					data_list.push_back(learning_rate[j]->at(i + testing_layer_count));
				std::vector<const_cuda_linear_buffer_device_smart_ptr> device_data = updater_list[i]->get_learning_rate(data_list);
				res.push_back(device_data);
			}

			return res;
		}

		void network_updater_cuda::read_data(
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data_list,
			std::vector<network_data_smart_ptr>& res,
			cudaStream_t stream_id) const
		{
			const network_data_smart_ptr& first_data = res.front();
			unsigned int layer_id = 0;
			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator src_it = data_list.begin(); src_it != data_list.end(); ++src_it, ++layer_id)
			{
				std::vector<layer_data_smart_ptr> host_data_list;
				for(std::vector<network_data_smart_ptr>::const_iterator sample_it = res.begin(); sample_it != res.end(); sample_it++)
					host_data_list.push_back((*sample_it)->at(layer_id + testing_layer_count));
				updater_list[layer_id]->get_data_from_device(*src_it, host_data_list);
			}
		}

		void network_updater_cuda::update_buffers_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = testing_schema_data.begin(); it != testing_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_tester_cuda_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = updater_schema_data.begin(); it != updater_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_updater_cuda_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration, updater_entry_count);

			for(std::map<unsigned int, weight_vector_bound_cuda_smart_ptr>::const_iterator it = weight_vector_bounds.begin(); it != weight_vector_bounds.end(); ++it)
				it->second->update_buffer_configuration(buffer_configuration, updater_entry_count);
		}

		unsigned int network_updater_cuda::get_max_batch_size() const
		{
			buffer_cuda_size_configuration buffer_configuration;

			for(std::vector<layer_updater_cuda_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);

			for(std::map<unsigned int, weight_vector_bound_cuda_smart_ptr>::const_iterator it = weight_vector_bounds.begin(); it != weight_vector_bounds.end(); ++it)
				it->second->update_buffer_configuration(buffer_configuration);

			return cuda_config->get_max_entry_count(buffer_configuration, 0.5F);
		}

		void network_updater_cuda::enqueue_dropout(
			cudaStream_t stream_id,
			const_cuda_linear_buffer_device_smart_ptr random_buffer,
			cuda_linear_buffer_device_smart_ptr target_buffer,
			float dropout_rate,
			unsigned int mask,
			unsigned int elem_count,
			unsigned int offset_in_random_list)
		{
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			dropout_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*target_buffer,
				*random_buffer,
				dropout_rate,
				offset_in_random_list,
				mask,
				elem_count);
		}
	}
}
