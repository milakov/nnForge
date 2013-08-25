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
#include "util_cuda.h"
#include "cuda_event.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

__global__ void convert_compacted_to_raw_kernel(
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

namespace nnforge
{
	namespace cuda
	{
		network_tester_cuda::network_tester_cuda(
			network_schema_smart_ptr schema,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_tester(schema)
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
		}

		void network_tester_cuda::update_buffers_configuration_testing(buffer_cuda_size_configuration& buffer_configuration) const
		{
			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = schema_data.begin(); it != schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_tester_cuda_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);
		}

		output_neuron_value_set_smart_ptr network_tester_cuda::actual_run(unsupervised_data_reader& reader)
		{
			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int output_neuron_count = (layer_config_list.end() - 1)->get_neuron_count();
			unsigned int entry_count = reader.get_entry_count();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			output_neuron_value_set_smart_ptr predicted_output_neuron_value_set(new output_neuron_value_set(entry_count, output_neuron_count));

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration_testing(buffers_config);

			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output

			unsigned int max_entry_count = std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), entry_count);

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

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_count);
				input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * input_neuron_elem_size));
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
					if (type_code == neuron_data_type::type_byte)
					{
						int elem_count = (input_neuron_count * entries_available_for_processing_count + 3) / 4;
						std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							elem_count);
						convert_compacted_to_raw_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
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
					else throw neural_network_exception((boost::format("actual_run cannot handle input neurons of type %1%") % type_code).str());

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
						bool entry_read = reader.read(input + (input_neuron_count * entries_read_count * input_neuron_elem_size));

						if (!entry_read)
							break;

						entries_read_count++;
					}
					cuda_safe_call(cudaMemcpyAsync(
						*(input_buf[current_data_slot]),
						input,
						entries_read_count * input_neuron_count * input_neuron_elem_size,
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

		std::vector<layer_configuration_specific_snapshot_smart_ptr> network_tester_cuda::actual_get_snapshot(
			const void * input,
			neuron_data_type::input_type type_code)
		{
			std::vector<layer_configuration_specific_snapshot_smart_ptr> res;

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int output_neuron_count = (layer_config_list.end() - 1)->get_neuron_count();
			size_t input_neuron_elem_size = neuron_data_type::get_input_size(type_code);

			cuda_linear_buffer_device_smart_ptr input_buf(new cuda_linear_buffer_device(input_neuron_count * input_neuron_elem_size));
			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			std::vector<cuda_linear_buffer_device_smart_ptr> output_buffer_list;
			output_buffer_list.push_back(output_buffer);
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(1);
				input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
				output_buffer_list.push_back(output_buffer);
			}

			// Copy input
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_buf,
					input,
					input_neuron_count * input_neuron_elem_size,
					cudaMemcpyHostToDevice,
					*command_stream));
			}

			// Convert input
			if (type_code == neuron_data_type::type_byte)
			{
				int elem_count = (input_neuron_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);
				convert_compacted_to_raw_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
					*input_buf,
					*input_converted_buf,
					elem_count);
			}
			else if (type_code == neuron_data_type::type_float)
			{
				cuda_safe_call(cudaMemcpyAsync(*input_converted_buf, *input_buf, input_neuron_count * sizeof(float), cudaMemcpyDeviceToDevice, *command_stream));
			}
			else throw neural_network_exception((boost::format("actual_get_snapshot cannot handle input neurons of type %1%") % type_code).str());

			{
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

		layer_configuration_specific_snapshot_smart_ptr network_tester_cuda::actual_run(
			const void * input,
			neuron_data_type::input_type type_code)
		{
			layer_configuration_specific_snapshot_smart_ptr res(new layer_configuration_specific_snapshot(layer_config_list[layer_config_list.size() - 1]));

			unsigned int input_neuron_count = layer_config_list.begin()->get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = layer_config_list.begin()->get_neuron_count_per_feature_map();
			unsigned int output_neuron_count = (layer_config_list.end() - 1)->get_neuron_count();
			size_t input_neuron_elem_size = neuron_data_type::get_input_size(type_code);

			cuda_linear_buffer_device_smart_ptr input_buf(new cuda_linear_buffer_device(input_neuron_count * input_neuron_elem_size));
			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(1);
				input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}

			// Copy input
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_buf,
					input,
					input_neuron_count * input_neuron_elem_size,
					cudaMemcpyHostToDevice,
					*command_stream));
			}

			// Convert input
			if (type_code == neuron_data_type::type_byte)
			{
				int elem_count = (input_neuron_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);
				convert_compacted_to_raw_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
					*input_buf,
					*input_converted_buf,
					elem_count);
			}
			else if (type_code == neuron_data_type::type_float)
			{
				cuda_safe_call(cudaMemcpyAsync(*input_converted_buf, *input_buf, input_neuron_count * sizeof(float), cudaMemcpyDeviceToDevice, *command_stream));
			}
			else throw neural_network_exception((boost::format("actual_run cannot handle input neurons of type %1%") % type_code).str());

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
