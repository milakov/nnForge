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

#include "layer_tester_cuda.h"

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		layer_tester_cuda::layer_tester_cuda()
		{
		}

		layer_tester_cuda::~layer_tester_cuda()
		{
		}

		void layer_tester_cuda::configure(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			const_layer_smart_ptr layer_schema,
			cuda_running_configuration_const_smart_ptr cuda_config)
		{
			this->layer_schema = layer_schema;
			this->input_configuration_specific = input_configuration_specific;
			this->output_configuration_specific = output_configuration_specific;
			this->cuda_config = cuda_config;

			input_elem_count_per_entry = input_configuration_specific.get_neuron_count();
			output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			input_elem_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();

			tester_configured();
		}

		void layer_tester_cuda::tester_configured()
		{
		}

		cuda_linear_buffer_device_smart_ptr layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return input_buffer;
		}

		std::vector<size_t> layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			return std::vector<size_t>();
		}

		std::vector<unsigned int> layer_tester_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}

		void layer_tester_cuda::update_buffer_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int tiling_factor) const
		{
			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it * tiling_factor);

			std::vector<size_t> fixed_sized = get_sizes_of_additional_buffers_fixed();
			for(std::vector<size_t>::const_iterator it = fixed_sized.begin(); it != fixed_sized.end(); ++it)
				buffer_configuration.add_constant_buffer(*it);

			std::vector<unsigned int> tex_per_entry = get_linear_addressing_through_texture_per_entry();
			for(std::vector<unsigned int>::const_iterator it = tex_per_entry.begin(); it != tex_per_entry.end(); ++it)
				buffer_configuration.add_per_entry_linear_addressing_through_texture(*it * tiling_factor);
		}

		std::vector<cuda_linear_buffer_device_smart_ptr> layer_tester_cuda::allocate_additional_buffers(unsigned int max_entry_count) const
		{
			std::vector<cuda_linear_buffer_device_smart_ptr> res;

			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
			{
				size_t sz = *it * max_entry_count;
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz)));
			}

			std::vector<size_t> fixed_sizes = get_sizes_of_additional_buffers_fixed();
			for(std::vector<size_t>::const_iterator it = fixed_sizes.begin(); it != fixed_sizes.end(); ++it)
			{
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(*it)));
			}

			fill_additional_buffers(res);

			return res;
		}

		std::vector<size_t> layer_tester_cuda::get_sizes_of_additional_buffers_fixed() const
		{
			return std::vector<size_t>();
		}

		void layer_tester_cuda::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
		}

		std::vector<const_cuda_linear_buffer_device_smart_ptr> layer_tester_cuda::get_data(const_layer_data_smart_ptr host_data) const
		{
			std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

			for(std::vector<std::vector<float> >::const_iterator it = host_data->begin(); it != host_data->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(float);
				cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		std::vector<const_cuda_linear_buffer_device_smart_ptr> layer_tester_cuda::set_get_data_custom(const_layer_data_custom_smart_ptr host_data_custom)
		{
			notify_data_custom(host_data_custom);

			std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

			for(std::vector<std::vector<int> >::const_iterator it = host_data_custom->begin(); it != host_data_custom->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(int);
				cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		void layer_tester_cuda::notify_data_custom(const_layer_data_custom_smart_ptr host_data_custom)
		{
		}
	}
}
