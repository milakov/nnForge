/*
 *  Copyright 2011-2016 Maxim Milakov
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
		void layer_tester_cuda::configure(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			layer::const_ptr layer_schema,
			cuda_running_configuration::const_ptr cuda_config)
		{
			this->layer_schema = layer_schema;
			this->input_configuration_specific_list = input_configuration_specific_list;
			this->output_configuration_specific = output_configuration_specific;
			this->cuda_config = cuda_config;

			input_elem_count_per_entry_list.resize(input_configuration_specific_list.size());
			input_elem_count_per_feature_map_list.resize(input_configuration_specific_list.size());
			for(int i = 0; i < input_configuration_specific_list.size(); ++i)
			{
				input_elem_count_per_entry_list[i] = input_configuration_specific_list[i].get_neuron_count();
				input_elem_count_per_feature_map_list[i] = input_configuration_specific_list[i].get_neuron_count_per_feature_map();
			}

			output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();

			tester_configured();
		}

		void layer_tester_cuda::tester_configured()
		{
		}

		int layer_tester_cuda::get_input_index_layer_can_write() const
		{
			return -1;
		}

		std::vector<unsigned int> layer_tester_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_tester_cuda::get_data(layer_data::const_ptr host_data) const
		{
			std::vector<cuda_linear_buffer_device::const_ptr> res;

			for(std::vector<std::vector<float> >::const_iterator it = host_data->begin(); it != host_data->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(float);
				cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_tester_cuda::set_get_data_custom(layer_data_custom::const_ptr host_data_custom)
		{
			notify_data_custom(host_data_custom);

			std::vector<cuda_linear_buffer_device::const_ptr> res;

			for(std::vector<std::vector<int> >::const_iterator it = host_data_custom->begin(); it != host_data_custom->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(int);
				cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_tester_cuda::get_persistent_working_data() const
		{
			return std::vector<cuda_linear_buffer_device::const_ptr>();
		}

		std::pair<size_t, bool> layer_tester_cuda::get_temporary_working_fixed_buffer_size() const
		{
			return std::make_pair(0, false);
		}

		size_t layer_tester_cuda::get_temporary_working_per_entry_buffer_size() const
		{
			return 0;
		}

		void layer_tester_cuda::notify_data_custom(layer_data_custom::const_ptr host_data_custom)
		{
		}
	}
}
