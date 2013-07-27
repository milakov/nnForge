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

#include "hessian_calculator_cuda.h"

#include "neural_network_cuda_exception.h"
#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "util_cuda.h"
#include "cuda_event.h"

#include "layer_hessian_schema_factory.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

__global__ void convert_compacted_to_raw_hess_kernel(
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

__global__ void apply_average_to_hessian_sum_kernel(
	float * __restrict hessians,
	float elem_count_multiplier,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
		hessians[elem_id] *= elem_count_multiplier;
}

namespace nnforge
{
	namespace cuda
	{
		hessian_calculator_cuda::hessian_calculator_cuda(
			network_schema_smart_ptr schema,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: hessian_calculator(schema)
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
				hessian_schemas.push_back(single_layer_hessian_schema_factory::get_const_instance().create_hessian_schema_layer(*it, cuda_config));

			setup_network_cuda();

			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
				testing_schema_data.push_back((*it)->get_schema_buffers());

			for(const_layer_hessian_schema_list::const_iterator it = hessian_schemas.begin(); it != hessian_schemas.end(); ++it)
				hessian_schema_data.push_back((*it)->get_schema_buffers());
		}

		hessian_calculator_cuda::~hessian_calculator_cuda()
		{
		}

		void hessian_calculator_cuda::setup_network_cuda()
		{
			command_stream = cuda_stream_smart_ptr(new cuda_stream());
			data_stream = cuda_stream_smart_ptr(new cuda_stream());
		}

		network_data_smart_ptr hessian_calculator_cuda::actual_get_hessian(
			supervised_data_reader& reader,
			network_data_smart_ptr data,
			unsigned int hessian_entry_to_process_count)
		{
			network_data_smart_ptr res(new network_data(*schema));

			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();
			layer_configuration_specific output_configuration = reader.get_output_configuration();

			unsigned int input_neuron_count = input_configuration.get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();
			unsigned int output_neuron_count = output_configuration.get_neuron_count();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > net_data = enqueue_get_data(data, *command_stream);
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > net_data_squared = enqueue_get_data_squared(net_data, *command_stream);
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > hessian_data = enqueue_get_hessian(data, *command_stream);

			buffer_cuda_size_configuration buffers_config;
			update_buffers_configuration(buffers_config);

			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
			buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
			buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // initial error

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data_squared.begin(); it != net_data_squared.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = hessian_data.begin(); it != hessian_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffers_config.add_constant_buffer((*it2)->get_size());

			unsigned int max_entry_count = std::min<unsigned int>(cuda_config->get_max_entry_count(buffers_config), hessian_entry_to_process_count);

			cuda_linear_buffer_device_smart_ptr input_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * input_neuron_elem_size)),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * input_neuron_elem_size)),
			};

			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * max_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr initial_error_buf(new cuda_linear_buffer_device(output_neuron_count * max_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > testing_input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_count);
				testing_input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_hessian_cuda::buffer_set> > hessian_input_and_all_buffers_pack;
			for(std::vector<layer_hessian_cuda_smart_ptr>::iterator it = hessian_list.begin(); it != hessian_list.end(); ++it)
			{
				layer_hessian_cuda::buffer_set all_buffers = (*it)->allocate_all_buffers(max_entry_count);
				hessian_input_and_all_buffers_pack.push_back(std::make_pair(output_buffer, all_buffers));
				output_buffer = all_buffers.output_neurons_buffer;
			}

			std::vector<cuda_linear_buffer_device_smart_ptr> output_errors_buffers;
			cuda_linear_buffer_device_smart_ptr output_errors = initial_error_buf;
			for(std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_hessian_cuda::buffer_set> >::reverse_iterator it = hessian_input_and_all_buffers_pack.rbegin(); it != hessian_input_and_all_buffers_pack.rend(); ++it)
			{
				output_errors_buffers.push_back(output_errors);
				layer_hessian_cuda::buffer_set& all_buffers = it->second;

				if (all_buffers.input_errors_buffer != 0)
					output_errors = all_buffers.input_errors_buffer;
			}

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_count * input_neuron_elem_size));
			unsigned char * input = *input_host_buf;

			unsigned int current_data_slot = 0;
			unsigned int current_command_slot = 1;
			unsigned int entries_available_for_copy_in_count = hessian_entry_to_process_count;
			unsigned int entries_available_for_processing_count = 0;
			cuda_event data_processed_event;
			cuda_event input_copied_event;
			if (cuda_config->is_flush_required())
			{
				cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
				cudaEventQuery(data_processed_event);
			}
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0))
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
						convert_compacted_to_raw_hess_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
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
					else throw neural_network_exception((boost::format("actual_get_hessian cannot handle input neurons of type %1%") % type_code).str());

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

					// Forward hessian
					{
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_hessian_cuda::buffer_set> >::iterator input_and_all_buffers_pack_it = hessian_input_and_all_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = hessian_schema_data.begin();
						for(std::vector<layer_hessian_cuda_smart_ptr>::iterator it = hessian_list.begin(); it != hessian_list.end(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++net_data_it)
						{
							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								*net_data_it,
								input_and_all_buffers_pack_it->first,
								input_and_all_buffers_pack_it->second.output_neurons_buffer,
								input_and_all_buffers_pack_it->second.additional_buffers,
								entries_available_for_processing_count);
						}
					}

					// Set initial errors to 1.0F
					{
						cuda_util::set_with_value(
							*cuda_config,
							*initial_error_buf,
							1.0F,
							output_neuron_count * entries_available_for_processing_count,
							*command_stream);
					}

					// Backward hessian
					{
						std::vector<cuda_linear_buffer_device_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin();
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_hessian_cuda::buffer_set> >::reverse_iterator input_and_all_buffers_pack_it = hessian_input_and_all_buffers_pack.rbegin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_squared_it = net_data_squared.rbegin();
						std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator hessian_data_it = hessian_data.rbegin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator schema_data_it = hessian_schema_data.rbegin();
						for(std::vector<layer_hessian_cuda_smart_ptr>::reverse_iterator it = hessian_list.rbegin(); it != hessian_list.rend(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++hessian_data_it, ++output_errors_it, ++net_data_squared_it)
						{
							if (it != (hessian_list.rend() - 1))
								(*it)->enqueue_backprop(
									*command_stream,
									*schema_data_it,
									*net_data_squared_it,
									input_and_all_buffers_pack_it->second.output_neurons_buffer,
									*output_errors_it,
									input_and_all_buffers_pack_it->second.input_errors_buffer,
									input_and_all_buffers_pack_it->second.additional_buffers,
									entries_available_for_processing_count);

							(*it)->enqueue_update_hessian(
								*command_stream,
								*schema_data_it,
								*hessian_data_it,
								*output_errors_it,
								input_and_all_buffers_pack_it->first,
								input_and_all_buffers_pack_it->second.additional_buffers,
								entries_available_for_processing_count);
						}
					}

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cudaEventQuery(data_processed_event);
					}
				}

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
				{
					unsigned int entries_to_read_count = std::min<unsigned int>(max_entry_count, entries_available_for_copy_in_count);
					while(entries_read_count < entries_to_read_count)
					{
						bool entry_read = reader.read(input + (input_neuron_count * entries_read_count * input_neuron_elem_size), 0);

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

				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;
			}

			enqueue_average_hessian(hessian_data, hessian_entry_to_process_count, *command_stream);
			enqueue_read_hessian(hessian_data, res, *command_stream);
			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			return res;
		}

		void hessian_calculator_cuda::layer_config_list_modified()
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

			hessian_list.clear();
			for(const_layer_hessian_schema_list::const_iterator it = hessian_schemas.begin(); it != hessian_schemas.end(); ++it, ++it_conf)
			{
				hessian_list.push_back(
					(*it)->create_hessian(
						*it_conf,
						*(it_conf + 1),
						(it_conf > layer_config_list.begin() + testing_layer_count)));
			}
		}

		std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > hessian_calculator_cuda::enqueue_get_data(
			network_data_smart_ptr data,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > res;

			for(layer_data_list::const_iterator it = data->begin() + testing_layer_count; it != data->end(); ++it)
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> buffer_list;
				const_layer_data_smart_ptr current_layer_data = *it;
				for(layer_data::const_iterator it2 = current_layer_data->begin(); it2 != current_layer_data->end(); ++it2)
				{
					const std::vector<float>& current_data = *it2;
					buffer_list.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*current_data.begin()),
						current_data.size() * sizeof(float),
						stream_id)));
				}
				res.push_back(buffer_list);
			}

			return res;
		}

		std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > hessian_calculator_cuda::enqueue_get_data_squared(
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > data,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > res;

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator it = data.begin(); it != data.end(); ++it)
			{
				std::vector<const_cuda_linear_buffer_device_smart_ptr> buffer_list;
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
				{
					const_cuda_linear_buffer_device_smart_ptr current_data = *it2;
					cuda_linear_buffer_device_smart_ptr new_buf(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(current_data->get_size())));

					cuda_util::multiply_by_itself(
						*cuda_config,
						*current_data,
						*new_buf,
						current_data->get_size() / sizeof(float),
						stream_id);

					buffer_list.push_back(new_buf);
				}
				res.push_back(buffer_list);
			}

			return res;
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > hessian_calculator_cuda::enqueue_get_hessian(
			network_data_smart_ptr data_use_schema_only,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			for(layer_data_list::const_iterator it = data_use_schema_only->begin() + testing_layer_count; it != data_use_schema_only->end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> buffer_list;
				const_layer_data_smart_ptr current_layer_data = *it;
				for(layer_data::const_iterator it2 = current_layer_data->begin(); it2 != current_layer_data->end(); ++it2)
				{
					const std::vector<float>& current_data = *it2;
					cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(current_data.size() * sizeof(float)));
					cuda_util::set_with_value(
						*cuda_config,
						*new_buf,
						0.0F,
						current_data.size(),
						stream_id);

					buffer_list.push_back(new_buf);
				}
				res.push_back(buffer_list);
			}

			return res;
		}

		void hessian_calculator_cuda::enqueue_average_hessian(
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& hessian_data,
			float hessian_entry_to_process_count,
			cudaStream_t stream_id) const
		{
			float mult = 1.0F / static_cast<float>(hessian_entry_to_process_count);
			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator it = hessian_data.begin(); it != hessian_data.end(); ++it)
			{
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
				{
					cuda_linear_buffer_device_smart_ptr buf = *it2;
					cuda_util::multiply_by_value(
						*cuda_config,
						*buf,
						mult,
						buf->get_size() / sizeof(float),
						stream_id);
				}
			}
		}

		void hessian_calculator_cuda::enqueue_read_hessian(
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& hessian_data,
			network_data_smart_ptr res,
			cudaStream_t stream_id) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator src_it = hessian_data.begin();
			for(layer_data_list::iterator it = res->begin() + testing_layer_count; it != res->end(); ++it, ++src_it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr>::iterator src_it2 = src_it->begin();
				for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2, ++src_it2)
				{
					std::vector<float>& dest = *it2;
					cuda_linear_buffer_device_smart_ptr src = *src_it2;
					cuda_safe_call(cudaMemcpyAsync(&(*dest.begin()), *src, dest.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_id));
				}
			}
		}

		void hessian_calculator_cuda::update_buffers_configuration(buffer_cuda_size_configuration& buffer_configuration) const
		{
			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = testing_schema_data.begin(); it != testing_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_tester_cuda_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = hessian_schema_data.begin(); it != hessian_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_hessian_cuda_smart_ptr>::const_iterator it = hessian_list.begin(); it != hessian_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);
		}
	}
}
