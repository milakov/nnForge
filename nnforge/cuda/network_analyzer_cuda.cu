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

#include "network_analyzer_cuda.h"

#include "neural_network_cuda_exception.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "util_cuda.h"
#include "cuda_event.h"
#include "layer_updater_schema_factory.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		__global__ void convert_compacted_to_raw_analazer_kernel(
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

		network_analyzer_cuda::network_analyzer_cuda(
			network_schema_smart_ptr schema,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_analyzer(schema)
			, cuda_config(cuda_config)
		{
			const const_layer_list& layer_list = *schema;

			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				updater_schemas.push_back(single_layer_updater_schema_factory::get_const_instance().create_updater_schema_layer(*it, cuda_config));

			setup_network_cuda();

			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it)
				updater_schema_data.push_back((*it)->get_schema_buffers());
		}

		network_analyzer_cuda::~network_analyzer_cuda()
		{
		}

		void network_analyzer_cuda::setup_network_cuda()
		{
			command_stream = cuda_stream_smart_ptr(new cuda_stream());
		}

		void network_analyzer_cuda::layer_config_list_modified()
		{
			updater_list.clear();
			updater_input_and_all_buffers_pack.clear();
			output_errors_buffers.clear();

			layer_configuration_specific_list::const_iterator it_conf = layer_config_list.begin();
			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it, ++it_conf)
			{
				updater_list.push_back(
					(*it)->create_updater(
						*it_conf,
						*(it_conf + 1),
						true));
			}

			unsigned int input_neuron_count = layer_config_list.front().get_neuron_count();
			unsigned int output_neuron_count = layer_config_list.back().get_neuron_count();
			input_buf = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));
			input_converted_buf = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it)
			{
				layer_updater_cuda::buffer_set all_buffers = (*it)->allocate_all_buffers(1);
				updater_input_and_all_buffers_pack.push_back(std::make_pair(output_buffer, all_buffers));
				output_buffer = all_buffers.output_neurons_buffer;
			}

			cuda_linear_buffer_device_smart_ptr initial_error_buf(new cuda_linear_buffer_device(output_neuron_count * sizeof(float)));
			cuda_linear_buffer_device_smart_ptr output_errors = initial_error_buf;
			for(std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator it = updater_input_and_all_buffers_pack.rbegin(); it != updater_input_and_all_buffers_pack.rend(); ++it)
			{
				output_errors_buffers.push_back(output_errors);
				layer_updater_cuda::buffer_set& all_buffers = it->second;

				if (all_buffers.input_errors_buffer != 0)
					output_errors = all_buffers.input_errors_buffer;
			}
		}

		void network_analyzer_cuda::actual_set_data(network_data_smart_ptr data)
		{
			net_data.clear();

			for(layer_data_list::const_iterator it2 = data->begin(); it2 != data->end(); ++it2)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> res;

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

		void network_analyzer_cuda::actual_set_input_data(
			const void * input,
			neuron_data_type::input_type type_code)
		{
			unsigned int input_neuron_count = layer_config_list.front().get_neuron_count();
			unsigned int output_neuron_count = layer_config_list.back().get_neuron_count();
			size_t input_neuron_elem_size = neuron_data_type::get_input_size(type_code);

			// Convert input
			if (type_code == neuron_data_type::type_byte)
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_buf,
					input,
					input_neuron_count * input_neuron_elem_size,
					cudaMemcpyHostToDevice,
					*command_stream));
				int elem_count = (input_neuron_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);
				convert_compacted_to_raw_analazer_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
					*input_buf,
					*input_converted_buf,
					elem_count);
			}
			else if (type_code == neuron_data_type::type_float)
			{
				cuda_safe_call(cudaMemcpyAsync(
					*input_converted_buf,
					input,
					input_neuron_count * input_neuron_elem_size,
					cudaMemcpyHostToDevice,
					*command_stream));
			}
			else throw neural_network_exception((boost::format("actual_set_input_data cannot handle input neurons of type %1%") % type_code).str());

			// Forward updater
			{
				std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.begin();
				std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = updater_schema_data.begin();
				layer_configuration_specific_list::const_iterator layer_config_it = layer_config_list.begin();
				for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++net_data_it, ++layer_config_it)
				{
					(*it)->enqueue_test(
						0,
						*command_stream,
						*schema_data_it,
						*net_data_it,
						input_and_all_buffers_pack_it->first,
						input_and_all_buffers_pack_it->second.output_neurons_buffer,
						input_and_all_buffers_pack_it->second.additional_buffers,
						input_and_all_buffers_pack_it->second.dynamic_memobjects,
						1);
				}
			}

			cuda_safe_call(cudaStreamSynchronize(*command_stream));
		}

		std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> network_analyzer_cuda::actual_run_backprop(
			const layer_configuration_specific_snapshot& output_data,
			const std::vector<unsigned int>& output_offset_list,
			unsigned int output_layer_id,
			const std::vector<std::pair<unsigned int, unsigned int> >& input_rectangle_borders)
		{
			std::vector<cuda_linear_buffer_device_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin() + (output_errors_buffers.size() - output_layer_id - 1);
			// Initialize output errors
			{
				float * dst = **output_errors_it;
				cuda_util::set_with_value(*cuda_config, dst, 0.0F, (*output_errors_it)->get_size() / sizeof(float), *command_stream);
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
							cuda_safe_call(cudaMemcpyAsync(dst + dst_fm_offset + output_config.get_pos(dst_offset_list), &(*src_it), sequential_copy_elem_count * sizeof(float), cudaMemcpyHostToDevice, *command_stream));
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

			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.rbegin() + (updater_input_and_all_buffers_pack.size() - output_layer_id - 1);
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator schema_data_it = updater_schema_data.rbegin() + (updater_schema_data.size() - output_layer_id - 1);
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_it = net_data.rbegin() + (net_data.size() - output_layer_id - 1);
			for(std::vector<layer_updater_cuda_smart_ptr>::reverse_iterator it = updater_list.rbegin() + (updater_list.size() - output_layer_id - 1); it != updater_list.rend(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++output_errors_it, ++net_data_it)
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
					1);
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

				cuda_linear_buffer_device_smart_ptr input_errors_buf = updater_input_and_all_buffers_pack.front().second.input_errors_buffer;
				if (input_errors_buf == 0)
					input_errors_buf = output_errors_buffers.back();
				float * src = *input_errors_buf;
				float * src_input_data = *input_converted_buf;
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
						cuda_safe_call(cudaMemcpyAsync(&(*dst_it), src + src_fm_offset + input_config.get_pos(src_offset_list), sequential_copy_elem_count * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));
						cuda_safe_call(cudaMemcpyAsync(&(*dst_input_data_it), src_input_data + src_fm_offset + input_config.get_pos(src_offset_list), sequential_copy_elem_count * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));

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

			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			return std::make_pair(res, input_data);
		}
	}
}
