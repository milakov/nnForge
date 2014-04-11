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

#include "weight_vector_bound_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		weight_vector_bound_cuda::weight_vector_bound_cuda()
		{
		}

		weight_vector_bound_cuda::~weight_vector_bound_cuda()
		{
		}

		nnforge_shared_ptr<weight_vector_bound_cuda> weight_vector_bound_cuda::create(
			const_layer_smart_ptr layer_schema,
			cuda_running_configuration_const_smart_ptr cuda_config) const
		{
			nnforge_shared_ptr<weight_vector_bound_cuda> res = create_specific();

			res->layer_schema = layer_schema;
			res->cuda_config = cuda_config;

			res->weight_vector_bound_configured();

			return res;
		}

		void weight_vector_bound_cuda::weight_vector_bound_configured()
		{
		}

		void weight_vector_bound_cuda::update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const
		{
			std::vector<size_t> per_entry_sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = per_entry_sizes.begin(); it != per_entry_sizes.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			std::vector<unsigned int> tex_per_entry = get_linear_addressing_through_texture_per_entry();
			for(std::vector<unsigned int>::const_iterator it = tex_per_entry.begin(); it != tex_per_entry.end(); ++it)
				buffer_configuration.add_per_entry_linear_addressing_through_texture(*it);
		}

		void weight_vector_bound_cuda::update_buffer_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			std::vector<size_t> per_entry_sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = per_entry_sizes.begin(); it != per_entry_sizes.end(); ++it)
				buffer_configuration.add_constant_buffer(*it * updater_entry_count);
		}

		std::vector<cuda_linear_buffer_device_smart_ptr> weight_vector_bound_cuda::allocate_additional_buffers(unsigned int max_entry_count)
		{
			std::vector<cuda_linear_buffer_device_smart_ptr> res;

			std::vector<size_t> per_entry_sizes = get_sizes_of_additional_buffers_per_entry();
			for(std::vector<size_t>::const_iterator it = per_entry_sizes.begin(); it != per_entry_sizes.end(); ++it)
				res.push_back(cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(*it * max_entry_count)));

			return res;
		}

		std::vector<size_t> weight_vector_bound_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			return std::vector<size_t>();
		}

		std::vector<unsigned int> weight_vector_bound_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}
	}
}
