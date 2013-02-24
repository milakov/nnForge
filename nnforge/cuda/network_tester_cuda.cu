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

#include "network_tester_cuda.h"

#include "neural_network_cuda_exception.h"
#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "cuda_util.h"
#include "cuda_event.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

__global__ void convert_compacted_to_raw_kernel(
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

namespace nnforge
{
	namespace cuda
	{
		network_tester_cuda::network_tester_cuda(
			network_schema_smart_ptr schema,
			const_data_scale_params_smart_ptr scale_params,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_tester(schema, scale_params)
			, cuda_config(cuda_config)
		{
			const const_layer_list& layer_list = *schema;
			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
			{
				testing_schemas.push_back(single_layer_testing_schema_factory::get_const_instance().create_testing_schema_layer(*it, cuda_config));
			}

			setup_network_cuda();

			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
			{
				schema_data.push_back((*it)->get_schema_buffers());
			}
		}

		network_tester_cuda::~network_tester_cuda()
		{
		}

		void network_tester_cuda::setup_network_cuda()
		{
			command_stream = cuda_stream_smart_ptr(new cuda_stream());
			data_stream = cuda_stream_smart_ptr(new cuda_stream());
		}

		// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
		void network_tester_cuda::actual_set_data(network_data_smart_ptr data)
		{
			net_data.clear();

			for(layer_data_list::const_iterator it2 = data->begin(); it2 != data->end(); ++it2)
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

				for(std::vector<std::vector<float> >::iterator it = (*it2)->begin(); it != (*it2)->end(); ++it)
				{
					size_t buffer_size = it->size() * sizeof(float);
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
					cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
					res.push_back(new_buf);
				}

				net_data.push_back(res);
			}
		}

		// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
		// The layer_config_list is guaranteed to be compatible with schema
		void network_tester_cuda::layer_config_list_modified()
		{
			tester_list.clear();

			layer_configuration_specific_list::const_iterator it_conf = layer_config_list.begin();
			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it, ++it_conf)
			{
				tester_list.push_back(
					(*it)->create_tester(
						*it_conf,
						*(it_conf + 1)));
			}

			scale_multiplication = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
				&(*current_scale_params->multiplication_list.begin()),
				current_scale_params->multiplication_list.size() * sizeof(float)));

			scale_addition = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
				&(*current_scale_params->addition_list.begin()),
				current_scale_params->addition_list.size() * sizeof(float)));
		}

		void network_tester_cuda::update_buffers_configuration_testing(buffer_cuda_size_configuration& buffer_configuration) const
		{
			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());
			buffer_configuration.add_constant_buffer(scale_addition->get_size());
			buffer_configuration.add_constant_buffer(scale_multiplication->get_size());

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = schema_data.begin(); it != schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_tester_cuda_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);
		}

		void network_tester_cuda::actual_test(
			supervised_data_reader_byte& reader,
			testing_complete_result_set& result)
		{
			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();
			layer_configuration_specific output_configuration = reader.get_output_configuration();

			unsigned int input_neuron_count = input_configuration.get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();
			unsigned int output_neuron_count = output_configuration.get_neuron_count();
			unsigned int input_feature_map_count = input_configuration.feature_map_count;

			unsigned int entry_count = reader.get_entry_count();

			result.mse = testing_result_smart_ptr(new testing_result(output_neuron_count));

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration_testing(buffers_config);

			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output

			unsigned int max_entry_count = std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), entry_count);

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

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_count);
				input_and_additional_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> >(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * sizeof(unsigned char)));
			unsigned char * input = *input_host_buf;
			cuda_linear_buffer_host_smart_ptr output_predicted_host_buf(new cuda_linear_buffer_host(output_neuron_count * max_entry_count * sizeof(float)));
			float * output_predicted = *output_predicted_host_buf;
			std::vector<float> output_actual[2] = 
			{
				std::vector<float>(output_neuron_count * max_entry_count),
				std::vector<float>(output_neuron_count * max_entry_count)
			};

			unsigned int current_data_slot = 0;
			unsigned int current_command_slot = 1;
			unsigned int entries_available_for_copy_in_count = entry_count;
			unsigned int entries_available_for_processing_count = 0;
			unsigned int entries_available_for_copy_out_count = 0;
			unsigned int entries_processed_count = 0;
			cuda_event output_copied_event;
			cuda_event data_processed_event;
			cuda_event input_copied_event;
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0) || (entries_available_for_copy_out_count > 0))
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
						convert_compacted_to_raw_kernel<<<convert_compacted_to_raw_2d_surf_kernel_dims.first, convert_compacted_to_raw_2d_surf_kernel_dims.second, 0, *command_stream>>>(
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
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = input_and_additional_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = schema_data.begin();
						for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++net_data_it, ++schema_data_it)
						{
							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								*net_data_it,
								input_and_additional_buffers_pack_it->first,
								input_and_additional_buffers_pack_it->second,
								entries_available_for_processing_count);
						}
					}

					// Copy output
					{
						cuda_safe_call(cudaMemcpyAsync(
							*output_buf[current_command_slot],
							*output_buffer,
							output_neuron_count * entries_available_for_processing_count * sizeof(float),
							cudaMemcpyDeviceToDevice,
							*command_stream));
					}

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cudaEventQuery(data_processed_event);
					}
				}

				if (entries_available_for_copy_out_count > 0)
				{
					cuda_safe_call(cudaMemcpyAsync(
						output_predicted,
						*(output_buf[current_data_slot]),
						entries_available_for_copy_out_count * output_neuron_count * sizeof(float),
						cudaMemcpyDeviceToHost,
						*data_stream));
					cuda_safe_call(cudaStreamSynchronize(*data_stream));

					const float * predicted_it = output_predicted;
					std::vector<float>::const_iterator actual_it = output_actual[current_data_slot].begin();
					const std::vector<float>::iterator mse_it_begin = result.mse->cumulative_mse_list.begin();

					for(std::vector<std::vector<float> >::iterator it = result.predicted_output_neuron_value_set->neuron_value_list.begin() + entries_processed_count;
						it != result.predicted_output_neuron_value_set->neuron_value_list.begin() + entries_processed_count + entries_available_for_copy_out_count;
						++it)
					{
						std::vector<float>& value_list = *it;
						std::copy(predicted_it, predicted_it + output_neuron_count, value_list.begin());

						std::vector<float>::const_iterator actual_itt = actual_it;
						const float * predicted_itt = predicted_it;
						for(std::vector<float>::iterator itt = mse_it_begin; itt != mse_it_begin + output_neuron_count; ++itt, ++actual_itt, ++predicted_itt)
						{
							float diff = *actual_itt - *predicted_itt;
							*itt += diff * diff * 0.5F;
						}

						predicted_it += output_neuron_count;
						actual_it += output_neuron_count;
					}
					
					entries_processed_count += entries_available_for_copy_out_count;
				}

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
				{
					while(entries_read_count < max_entry_count)
					{
						bool entry_read = reader.read(
							input + (input_neuron_count * entries_read_count),
							&(*output_actual[current_data_slot].begin()) + (output_neuron_count * entries_read_count));

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
				}

				cuda_safe_call(cudaStreamSynchronize(*data_stream));
				cuda_safe_call(cudaStreamSynchronize(*command_stream));

				entries_available_for_copy_out_count = entries_available_for_processing_count;
				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;
			}

			result.mse->entry_count = entries_processed_count;
		}

		output_neuron_value_set_smart_ptr network_tester_cuda::actual_run(unsupervised_data_reader_byte& reader)
		{
			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int input_feature_map_count = layer_config_list.begin()->feature_map_count;
			unsigned int output_neuron_count = layer_config_list.end()->get_neuron_count();

			unsigned int entry_count = reader.get_entry_count();

			output_neuron_value_set_smart_ptr predicted_output_neuron_value_set(new output_neuron_value_set(entry_count, output_neuron_count));

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration_testing(buffers_config);

			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(unsigned char)); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output

			unsigned int max_entry_count = std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), entry_count);

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

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_count);
				input_and_additional_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> >(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * sizeof(unsigned char)));
			unsigned char * input = *input_host_buf;
			cuda_linear_buffer_host_smart_ptr output_predicted_host_buf(new cuda_linear_buffer_host(output_neuron_count * max_entry_count * sizeof(float)));
			float * output_predicted = *output_predicted_host_buf;

			unsigned int current_data_slot = 0;
			unsigned int current_command_slot = 1;
			unsigned int entries_available_for_copy_in_count = entry_count;
			unsigned int entries_available_for_processing_count = 0;
			unsigned int entries_available_for_copy_out_count = 0;
			unsigned int entries_processed_count = 0;
			cuda_event output_copied_event;
			cuda_event data_processed_event;
			cuda_event input_copied_event;
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0) || (entries_available_for_copy_out_count > 0))
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
						convert_compacted_to_raw_kernel<<<convert_compacted_to_raw_2d_surf_kernel_dims.first, convert_compacted_to_raw_2d_surf_kernel_dims.second, 0, *command_stream>>>(
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
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = input_and_additional_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = schema_data.begin();
						for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++net_data_it, ++schema_data_it)
						{
							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								*net_data_it,
								input_and_additional_buffers_pack_it->first,
								input_and_additional_buffers_pack_it->second,
								entries_available_for_processing_count);
						}
					}

					// Copy output
					{
						cuda_safe_call(cudaMemcpyAsync(
							*output_buf[current_command_slot],
							*output_buffer,
							output_neuron_count * entries_available_for_processing_count * sizeof(float),
							cudaMemcpyDeviceToDevice,
							*command_stream));
					}

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cudaEventQuery(data_processed_event);
					}
				}

				if (entries_available_for_copy_out_count > 0)
				{
					cuda_safe_call(cudaMemcpyAsync(
						output_predicted,
						*(output_buf[current_data_slot]),
						entries_available_for_copy_out_count * output_neuron_count * sizeof(float),
						cudaMemcpyDeviceToHost,
						*data_stream));
					cuda_safe_call(cudaStreamSynchronize(*data_stream));

					const float * predicted_it = output_predicted;
					for(std::vector<std::vector<float> >::iterator it = predicted_output_neuron_value_set->neuron_value_list.begin() + entries_processed_count;
						it != predicted_output_neuron_value_set->neuron_value_list.begin() + entries_processed_count + entries_available_for_copy_out_count;
						it++, predicted_it += output_neuron_count)
					{
						std::vector<float>& value_list = *it;
						std::copy(predicted_it, predicted_it + output_neuron_count, value_list.begin());
					}
					
					entries_processed_count += entries_available_for_copy_out_count;
				}

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
				{
					while(entries_read_count < max_entry_count)
					{
						bool entry_read = reader.read(input + (input_neuron_count * entries_read_count));

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
				}

				cuda_safe_call(cudaStreamSynchronize(*data_stream));
				cuda_safe_call(cudaStreamSynchronize(*command_stream));

				entries_available_for_copy_out_count = entries_available_for_processing_count;
				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;
			}


			return predicted_output_neuron_value_set;
		}

		std::vector<layer_configuration_specific_snapshot_smart_ptr> network_tester_cuda::actual_get_snapshot(std::vector<unsigned char>& input)
		{
			std::vector<layer_configuration_specific_snapshot_smart_ptr> res;

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int input_feature_map_count = layer_config_list.begin()->feature_map_count;
			unsigned int output_neuron_count = (layer_config_list.end() - 1)->get_neuron_count();

			cuda_linear_buffer_device_smart_ptr input_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(unsigned char)));
			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			std::vector<cuda_linear_buffer_device_smart_ptr> output_buffer_list;
			output_buffer_list.push_back(output_buffer);
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(1);
				input_and_additional_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> >(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				output_buffer_list.push_back(output_buffer);
			}

			// Copy inout
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_buf,
					&(*input.begin()),
					input_neuron_count * sizeof(unsigned char),
					cudaMemcpyHostToDevice,
					*command_stream));
			}

			// Convert input
			{
				std::pair<dim3, dim3> convert_compacted_to_raw_2d_surf_kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					input_neuron_count_per_feature_map,
					input_feature_map_count,
					1);
				convert_compacted_to_raw_kernel<<<convert_compacted_to_raw_2d_surf_kernel_dims.first, convert_compacted_to_raw_2d_surf_kernel_dims.second, 0, *command_stream>>>(
					*input_buf,
					*input_converted_buf,
					*scale_addition,
					*scale_multiplication,
					input_neuron_count_per_feature_map,
					input_feature_map_count,
					1);

				layer_configuration_specific_snapshot_smart_ptr input_elem(new layer_configuration_specific_snapshot(layer_config_list[0]));
				res.push_back(input_elem);

				cuda_safe_call(cudaMemcpyAsync(
					&(*(input_elem->data.begin())),
					*output_buffer_list[0],
					input_elem->data.size() * sizeof(float),
					cudaMemcpyDeviceToHost,
					*command_stream));
			}

			// Run ann
			{
				std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = input_and_additional_buffers_pack.begin();
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = schema_data.begin();
				int layer_id = 0;
				int output_buffer_id = 1;
				for(
					std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin();
					it != tester_list.end();
					++it, ++input_and_additional_buffers_pack_it, ++net_data_it, ++schema_data_it, ++layer_id, ++output_buffer_id)
				{
					(*it)->enqueue_test(
						*command_stream,
						*schema_data_it,
						*net_data_it,
						input_and_additional_buffers_pack_it->first,
						input_and_additional_buffers_pack_it->second,
						1);

					layer_configuration_specific_snapshot_smart_ptr new_elem(new layer_configuration_specific_snapshot(layer_config_list[layer_id + 1]));
					res.push_back(new_elem);

					cuda_safe_call(cudaMemcpyAsync(
						&(*(new_elem->data.begin())),
						*output_buffer_list[output_buffer_id],
						new_elem->data.size() * sizeof(float),
						cudaMemcpyDeviceToHost,
						*command_stream));
				}
			}

			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			return res;
		}

		layer_configuration_specific_snapshot_smart_ptr network_tester_cuda::actual_run(std::vector<unsigned char>& input)
		{
			layer_configuration_specific_snapshot_smart_ptr res(new layer_configuration_specific_snapshot(layer_config_list[layer_config_list.size() - 1]));

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int input_feature_map_count = layer_config_list.begin()->feature_map_count;
			unsigned int output_neuron_count = (layer_config_list.end() - 1)->get_neuron_count();

			cuda_linear_buffer_device_smart_ptr input_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(unsigned char)));
			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(1);
				input_and_additional_buffers_pack.push_back(std::make_pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> >(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}

			// Copy inout
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_buf,
					&(*input.begin()),
					input_neuron_count * sizeof(unsigned char),
					cudaMemcpyHostToDevice,
					*command_stream));
			}

			// Convert input
			{
				std::pair<dim3, dim3> convert_compacted_to_raw_2d_surf_kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					input_neuron_count_per_feature_map,
					input_feature_map_count,
					1);
				convert_compacted_to_raw_kernel<<<convert_compacted_to_raw_2d_surf_kernel_dims.first, convert_compacted_to_raw_2d_surf_kernel_dims.second, 0, *command_stream>>>(
					*input_buf,
					*input_converted_buf,
					*scale_addition,
					*scale_multiplication,
					input_neuron_count_per_feature_map,
					input_feature_map_count,
					1);
			}

			// Run ann
			{
				std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = input_and_additional_buffers_pack.begin();
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = schema_data.begin();
				for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++net_data_it, ++schema_data_it)
				{
					(*it)->enqueue_test(
						*command_stream,
						*schema_data_it,
						*net_data_it,
						input_and_additional_buffers_pack_it->first,
						input_and_additional_buffers_pack_it->second,
						1);
				}
			}

			// Copy output
			{
				cuda_safe_call(cudaMemcpyAsync(
					&(*(res->data.begin())),
					*output_buffer,
					output_neuron_count * sizeof(float),
					cudaMemcpyDeviceToHost,
					*command_stream));
			}

			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			return res;
		}
	}
}
