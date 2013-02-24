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

#include "network_updater_cuda.h"

#include "neural_network_cuda_exception.h"
#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "cuda_util.h"
#include "cuda_event.h"
#include "layer_updater_schema_factory.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>
#include <stack>

__global__ void convert_compacted_to_raw_upd_kernel(
	const unsigned char * __restrict input,
	float * __restrict output,
	const float * __restrict scale_addition,
	const float * __restrict scale_multiplication,
	int elem_count_per_feature_map,
	int feature_map_count,
	int entry_count)
{
	int elem_id_inside_feature_map = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
	int entry_id = blockIdx.z * blockDim.z + threadIdx.z;
	bool in_bounds = (entry_id < entry_count) && (elem_id_inside_feature_map < elem_count_per_feature_map) && (feature_map_id < feature_map_count);
	if (in_bounds)
	{
		int offset = elem_count_per_feature_map * (entry_id * feature_map_count + feature_map_id) + elem_id_inside_feature_map;
		unsigned char val = input[offset];
		float converted_val = ((val * (1.0F / 255.0F)) + scale_addition[feature_map_id]) * scale_multiplication[feature_map_id];
		output[offset] = converted_val;
	}
}

__global__ void compute_error_upd_kernel(
	float * __restrict errors,
	float * __restrict mse,
	const float * __restrict desired_output_neurons,
	const float * __restrict actual_output_neurons,
	int output_entry_id,
	int output_elem_count,
	int updater_entry_count)
{
	int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
	int updater_entry_id = blockIdx.y * blockDim.y + threadIdx.y;
	bool in_bounds = (elem_id < output_elem_count) && (updater_entry_id < updater_entry_count);
	if (in_bounds)
	{
		int offset = updater_entry_id * output_elem_count + elem_id;
		float err = desired_output_neurons[output_entry_id * output_elem_count + elem_id] - actual_output_neurons[offset];
		errors[offset] = err;
		mse[offset] += err * err * 0.5F;
	}
}

namespace nnforge
{
	namespace cuda
	{
		unsigned int network_updater_cuda::max_entry_count_in_single_batch = 1024;

		network_updater_cuda::network_updater_cuda(
			network_schema_smart_ptr schema,
			const_data_scale_params_smart_ptr scale_params,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_updater(schema, scale_params)
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
			supervised_data_reader_byte& reader,
			const std::vector<network_data_smart_ptr>& training_speed_vector_list,
			std::vector<network_data_smart_ptr>& data_list,
			const std::map<unsigned int, float>& layer_to_dropout_rate_map,
			const std::vector<float>& random_uniform_list)
		{
			std::vector<testing_result_smart_ptr> res;

			unsigned int min_dropout_layer_id = testing_layer_count + 1;
			for(std::map<unsigned int, float>::const_iterator it = layer_to_dropout_rate_map.begin(); it != layer_to_dropout_rate_map.end(); ++it)
				if (it->first < min_dropout_layer_id)
					throw neural_network_exception((boost::format("Unable to apply dropout to layer %1%") % it->first).str());

			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();
			layer_configuration_specific output_configuration = reader.get_output_configuration();

			unsigned int input_neuron_count = input_configuration.get_neuron_count();
			unsigned int output_neuron_count = output_configuration.get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();
			unsigned int input_feature_map_count = input_configuration.feature_map_count;

			unsigned int updater_entry_count = static_cast<unsigned int>(data_list.size());
			if (updater_entry_count == 0)
				return res;

			for(unsigned int i = 0; i < training_speed_vector_list.size(); ++i)
				res.push_back(testing_result_smart_ptr(new testing_result(output_neuron_count)));

			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > net_data = enqueue_get_data(data_list, *command_stream);
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > training_speed_data = enqueue_get_training_speed(training_speed_vector_list, *command_stream);

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration(buffers_config, updater_entry_count);

			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // initial error
			buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // mse
			if (!random_uniform_list.empty())
				buffers_config.add_constant_buffer(random_uniform_list.size() * sizeof(float)); // random_uniform_list

			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = training_speed_data.begin(); it != training_speed_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			unsigned int max_entry_count = std::min<unsigned int>(std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), reader.get_entry_count()), max_entry_count_in_single_batch);

			cuda_linear_buffer_device_smart_ptr input_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * sizeof(unsigned char))),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * sizeof(unsigned char))),
			};

			cuda_linear_buffer_device_smart_ptr output_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_count * sizeof(float))),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_count * sizeof(float))),
			};

			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr initial_error_buf(new cuda_linear_buffer_device(output_neuron_count * updater_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr mse_buf(new cuda_linear_buffer_device(output_neuron_count * updater_entry_count * sizeof(float)));

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
				testing_input_and_additional_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> >(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> > updater_input_and_all_buffers_pack;
			for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it)
			{
				layer_updater_cuda::buffer_set all_buffers = (*it)->allocate_all_buffers(updater_entry_count);
				updater_input_and_all_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set>(output_buffer, all_buffers));
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

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * sizeof(unsigned char)));
			unsigned char * input = *input_host_buf;
			cuda_linear_buffer_host_smart_ptr output_host_buf(new cuda_linear_buffer_host(output_neuron_count * max_entry_count * sizeof(float)));
			float * output = *output_host_buf;

			// zero mse
			cuda_util::set_with_value(
				*cuda_config,
				*mse_buf,
				0.0F,
				output_neuron_count * updater_entry_count,
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
				cudaEventQuery(data_processed_event);
			}
			std::tr1::variate_generator<random_generator, std::tr1::uniform_int<unsigned int> > gen_random_offset(
				rnd::get_random_generator(),
				std::tr1::uniform_int<unsigned int>(0, static_cast<unsigned int>(random_uniform_list.size() - 1)));
			unsigned int mask = static_cast<unsigned int>(random_uniform_list.size() - 1);
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0))
			{
				if (entries_available_for_processing_count > 0)
				{
					// Convert input
					{
						std::pair<dim3, dim3> convert_compacted_to_raw_2d_surf_kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							input_neuron_count_per_feature_map,
							input_feature_map_count,
							entries_available_for_processing_count);
						convert_compacted_to_raw_upd_kernel<<<convert_compacted_to_raw_2d_surf_kernel_dims.first, convert_compacted_to_raw_2d_surf_kernel_dims.second, 0, *command_stream>>>(
							*input_buf[current_command_slot],
							*input_converted_buf,
							*scale_addition,
							*scale_multiplication,
							input_neuron_count_per_feature_map,
							input_feature_map_count,
							entries_available_for_processing_count);
					}

					// Run ann
					{
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = testing_input_and_additional_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = testing_schema_data.begin();
						for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++schema_data_it)
						{
							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								std::vector<const_cuda_linear_buffer_device_smart_ptr>(),
								input_and_additional_buffers_pack_it->first,
								input_and_additional_buffers_pack_it->second,
								entries_available_for_processing_count);
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
							for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++net_data_it, ++layer_id)
							{
								if (it != updater_list.begin())
								{
									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = gen_random_offset();
										offset_list.push(offset);
										(*it)->enqueue_forward_dropout(
											*command_stream,
											random_uniform_buf,
											input_and_all_buffers_pack_it->first,
											dropout_it->second,
											mask,
											updater_entry_count,
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
									updater_entry_count);
							}
						}

						// Compute errors
						{
							std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
								*cuda_config,
								output_neuron_count,
								updater_entry_count,
								1);
							compute_error_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
								*initial_error_buf,
								*mse_buf,
								*output_buf[current_command_slot],
								*output_buffer,
								input_entry_id,
								output_neuron_count,
								updater_entry_count);
						}

						// Backward updater
						{
							std::vector<cuda_linear_buffer_device_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin();
							std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.rbegin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_it = net_data.rbegin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator training_speed_data_it = training_speed_data.rbegin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator schema_data_it = updater_schema_data.rbegin();
							unsigned int reverse_layer_id = static_cast<unsigned int>(updater_list.size() + testing_layer_count) - 1;
							for(std::vector<layer_updater_cuda_smart_ptr>::reverse_iterator it = updater_list.rbegin(); it != updater_list.rend(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++training_speed_data_it, ++output_errors_it, ++net_data_it, --reverse_layer_id)
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
										updater_entry_count);

									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(reverse_layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = offset_list.top();
										offset_list.pop();
										(*it)->enqueue_backward_dropout(
											*command_stream,
											random_uniform_buf,
											input_and_all_buffers_pack_it->second.input_errors_buffer,
											dropout_it->second,
											mask,
											updater_entry_count,
											offset);
									}
								}

								(*it)->enqueue_update_weights(
									(it == (updater_list.rend() - 1)) ? input_entry_id : 0,
									*command_stream,
									*net_data_it,
									*schema_data_it,
									*training_speed_data_it,
									*output_errors_it,
									input_and_all_buffers_pack_it->first,
									input_and_all_buffers_pack_it->second.additional_buffers,
									updater_entry_count);
							}
						}

						if (((input_entry_id % 16) == 1) && cuda_config->is_flush_required())
						{
							cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
							cudaEventQuery(data_processed_event);
						}
					} // for(unsigned int input_entry_id

					if (profile_mode)
						entry_count_updated_in_profile_mode = entries_available_for_processing_count;

					for(std::vector<testing_result_smart_ptr>::iterator it = res.begin(); it != res.end(); ++it)
						(*it)->entry_count += entries_available_for_processing_count;

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cudaEventQuery(data_processed_event);
					}
				} // if (entries_available_for_processing_count > 0)

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
				{
					unsigned int entries_to_read_count = std::min<unsigned int>(max_entry_count, entries_available_for_copy_in_count);
					while(entries_read_count < entries_to_read_count)
					{
						bool entry_read = reader.read(
							input + (input_neuron_count * entries_read_count),
							output + (output_neuron_count * entries_read_count));

						if (!entry_read)
							break;

						entries_read_count++;
					}
					cuda_safe_call(cudaMemcpyAsync(
						*(input_buf[current_data_slot]),
						input,
						entries_read_count * input_neuron_count * sizeof(unsigned char),
						cudaMemcpyHostToDevice,
						*data_stream));
					cuda_safe_call(cudaMemcpyAsync(
						*(output_buf[current_data_slot]),
						output,
						entries_read_count * output_neuron_count * sizeof(float),
						cudaMemcpyHostToDevice,
						*data_stream));
				}

				cuda_safe_call(cudaStreamSynchronize(*data_stream));
				cuda_safe_call(cudaStreamSynchronize(*command_stream));

				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;

				if (profile_mode)
					entries_available_for_copy_in_count = 0;
			}

			read_data(net_data, data_list, *command_stream);

			std::vector<float> mse_list(output_neuron_count * updater_entry_count);
			cuda_safe_call(cudaMemcpyAsync(&(*mse_list.begin()), *mse_buf, mse_list.size() * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));
			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			for(unsigned int i = 0; i < updater_entry_count; ++i)
				std::copy(mse_list.begin() + output_neuron_count * i, mse_list.begin() + output_neuron_count * (i + 1), res[i]->cumulative_mse_list.begin());

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
			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it, ++it_conf)
			{
				updater_list.push_back(
					(*it)->create_updater(
						*it_conf,
						*(it_conf + 1),
						(it_conf > layer_config_list.begin() + testing_layer_count),
						(it_conf > layer_config_list.begin() + testing_layer_count)));
			}

			scale_multiplication = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
				&(*current_scale_params->multiplication_list.begin()),
				current_scale_params->multiplication_list.size() * sizeof(float)));

			scale_addition = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
				&(*current_scale_params->addition_list.begin()),
				current_scale_params->addition_list.size() * sizeof(float)));
		}

		std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::enqueue_get_training_speed(
			const std::vector<network_data_smart_ptr>& training_speed_list,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > res;

			const network_data_smart_ptr& first_data = training_speed_list[0];

			for(unsigned int layer_id = testing_layer_count; layer_id < updater_schemas.size() + testing_layer_count; ++layer_id)
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> buffer_list;
				unsigned int subindex = 0;
				for(std::vector<std::vector<float> >::iterator it = (*first_data)[layer_id]->begin(); it != (*first_data)[layer_id]->end(); ++it, ++subindex)
				{
					size_t single_size = it->size();
					std::vector<float> pack(single_size * training_speed_list.size());

					std::vector<float>::iterator fill_it = pack.begin();
					for(std::vector<network_data_smart_ptr>::const_iterator sample_it = training_speed_list.begin(); sample_it != training_speed_list.end(); sample_it++)
					{
						const std::vector<float>& inp_buf = (*sample_it)->at(layer_id)->at(subindex);
						fill_it = std::copy(inp_buf.begin(), inp_buf.end(), fill_it);
					}

					buffer_list.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*pack.begin()),
						pack.size() * sizeof(float),
						stream_id)));
				}
				res.push_back(buffer_list);
			}

			return res;
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::enqueue_get_data(
			const std::vector<network_data_smart_ptr>& data_list,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			const network_data_smart_ptr& first_data = data_list[0];

			for(unsigned int layer_id = testing_layer_count; layer_id < updater_schemas.size() + testing_layer_count; ++layer_id)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> buffer_list;
				unsigned int subindex = 0;
				for(std::vector<std::vector<float> >::iterator it = (*first_data)[layer_id]->begin(); it != (*first_data)[layer_id]->end(); ++it, ++subindex)
				{
					size_t single_size = it->size();
					std::vector<float> pack(single_size * data_list.size());

					std::vector<float>::iterator fill_it = pack.begin();
					for(std::vector<network_data_smart_ptr>::const_iterator sample_it = data_list.begin(); sample_it != data_list.end(); sample_it++)
					{
						const std::vector<float>& inp_buf = (*sample_it)->at(layer_id)->at(subindex);
						fill_it = std::copy(inp_buf.begin(), inp_buf.end(), fill_it);
					}

					buffer_list.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*pack.begin()),
						pack.size() * sizeof(float),
						stream_id)));
				}
				res.push_back(buffer_list);
			}

			return res;
		}

		void network_updater_cuda::read_data(
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data_list,
			std::vector<network_data_smart_ptr>& res,
			cudaStream_t stream_id) const
		{
			const network_data_smart_ptr& first_data = res[0];
			unsigned int layer_id = testing_layer_count;
			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator src_it = data_list.begin(); src_it != data_list.end(); ++src_it, ++layer_id)
			{
				unsigned int subindex = 0;
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::iterator src_it2 = src_it->begin(); src_it2 != src_it->end(); ++src_it2, ++subindex)
				{
					cuda_linear_buffer_device_smart_ptr src = *src_it2;
					std::vector<float> pack(src->get_size() / sizeof(float));
					cuda_safe_call(cudaMemcpyAsync(&(*pack.begin()), *src, pack.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_id));
					cuda_safe_call(cudaStreamSynchronize(stream_id));

					std::vector<float>::const_iterator src_buf_it = pack.begin();
					for(std::vector<network_data_smart_ptr>::const_iterator sample_it = res.begin(); sample_it != res.end(); sample_it++)
					{
						std::vector<float>& dst_buf = (*sample_it)->at(layer_id)->at(subindex);
						std::copy(src_buf_it, src_buf_it + dst_buf.size(), dst_buf.begin());
						src_buf_it += dst_buf.size();
					}
				}
			}
		}

		void network_updater_cuda::update_buffers_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			buffer_configuration.add_constant_buffer(scale_addition->get_size());
			buffer_configuration.add_constant_buffer(scale_multiplication->get_size());

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
		}

		unsigned int network_updater_cuda::get_max_batch_size() const
		{
			buffer_cuda_size_configuration buffer_configuration;

			for(std::vector<layer_updater_cuda_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);

			return cuda_config->get_max_entry_count(buffer_configuration, 0.5F);
		}
	}
}
