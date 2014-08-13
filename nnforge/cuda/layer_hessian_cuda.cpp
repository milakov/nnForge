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

#include "layer_hessian_cuda.h"

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		layer_hessian_cuda::layer_hessian_cuda()
		{
		}

		layer_hessian_cuda::~layer_hessian_cuda()
		{
		}

		void layer_hessian_cuda::configure(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			const_layer_smart_ptr layer_schema,
			cuda_running_configuration_const_smart_ptr cuda_config,
			bool backprop_required)
		{
			this->layer_schema = layer_schema;
			this->input_configuration_specific = input_configuration_specific;
			this->output_configuration_specific = output_configuration_specific;
			this->cuda_config = cuda_config;
			this->backprop_required = backprop_required;

			input_elem_count_per_entry = input_configuration_specific.get_neuron_count();
			output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			input_elem_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();

			hessian_configured();
		}

		void layer_hessian_cuda::hessian_configured()
		{
		}

		std::vector<size_t> layer_hessian_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			return std::vector<size_t>();
		}

		std::vector<unsigned int> layer_hessian_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}

		void layer_hessian_cuda::update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const
		{
			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			std::vector<size_t> fixed_sized = get_sizes_of_additional_buffers_fixed();
			for(std::vector<size_t>::const_iterator it = fixed_sized.begin(); it != fixed_sized.end(); ++it)
				buffer_configuration.add_constant_buffer(*it);

			buffer_configuration.add_per_entry_buffer(output_elem_count_per_entry * sizeof(float));

			if (backprop_required && !is_in_place_backprop())
				buffer_configuration.add_per_entry_buffer(input_elem_count_per_entry * sizeof(float));

			std::vector<unsigned int> tex_per_entry = get_linear_addressing_through_texture_per_entry();
			for(std::vector<unsigned int>::const_iterator it = tex_per_entry.begin(); it != tex_per_entry.end(); ++it)
				buffer_configuration.add_per_entry_linear_addressing_through_texture(*it);
		}

		layer_hessian_cuda::buffer_set layer_hessian_cuda::allocate_all_buffers(unsigned int max_entry_count) const
		{
			buffer_set res;

			std::vector<size_t> sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = sizes.begin(); it != sizes.end(); ++it)
			{
				size_t sz = *it * max_entry_count;
				res.additional_buffers.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz)));
			}

			std::vector<size_t> fixed_sizes = get_sizes_of_additional_buffers_fixed();
			for(std::vector<size_t>::const_iterator it = fixed_sizes.begin(); it != fixed_sizes.end(); ++it)
			{
				res.additional_buffers.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(*it)));
			}

			{
				size_t sz = output_elem_count_per_entry * sizeof(float) * max_entry_count;
				res.output_neurons_buffer = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz));
			}

			if (backprop_required && !is_in_place_backprop())
			{
				size_t sz = input_elem_count_per_entry * sizeof(float) * max_entry_count;
				res.input_errors_buffer = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(sz));
			}

			fill_additional_buffers(res.additional_buffers);

			return res;
		}

		void layer_hessian_cuda::enqueue_update_hessian(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& hessian_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
		}

		std::vector<size_t> layer_hessian_cuda::get_sizes_of_additional_buffers_fixed() const
		{
			return std::vector<size_t>();
		}

		void layer_hessian_cuda::fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const
		{
		}

		std::vector<const_cuda_linear_buffer_device_smart_ptr> layer_hessian_cuda::get_data(const_layer_data_smart_ptr host_data) const
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

		std::vector<const_cuda_linear_buffer_device_smart_ptr> layer_hessian_cuda::get_data_custom(const_layer_data_custom_smart_ptr host_data_custom) const
		{
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

		std::vector<const_cuda_linear_buffer_device_smart_ptr> layer_hessian_cuda::get_data_squared(const_layer_data_smart_ptr host_data) const
		{
			std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

			for(std::vector<std::vector<float> >::const_iterator it = host_data->begin(); it != host_data->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(float);
				cuda_linear_buffer_device_smart_ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				cuda_util::multiply_by_itself(
					*cuda_config,
					*new_buf,
					*new_buf,
					new_buf->get_size() / sizeof(float),
					0);
				res.push_back(new_buf);
			}
			cuda_safe_call(cudaStreamSynchronize(0));

			return res;
		}
	}
}
