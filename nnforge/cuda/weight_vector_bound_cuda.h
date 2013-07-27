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

#pragma once

#include "../layer.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_linear_buffer_device.h"

#include "../weight_vector_bound.h"

#include <map>

namespace nnforge
{
	namespace cuda
	{
		class weight_vector_bound_cuda
		{
		public:
			virtual ~weight_vector_bound_cuda();

			std::tr1::shared_ptr<weight_vector_bound_cuda> create(
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config) const;

			virtual const boost::uuids::uuid& get_uuid() const = 0;

			std::vector<cuda_linear_buffer_device_smart_ptr> allocate_additional_buffers(unsigned int max_entry_count);

			void update_buffer_configuration(buffer_cuda_size_configuration& buffer_configuration) const;

			void update_buffer_configuration(
				buffer_cuda_size_configuration& buffer_configuration,
				unsigned int updater_entry_count) const;

			virtual void enqueue_normalize_weights(
				cudaStream_t stream_id,
				const weight_vector_bound& bound,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

		protected:
			weight_vector_bound_cuda();

			virtual std::tr1::shared_ptr<weight_vector_bound_cuda> create_specific() const = 0;

			// The method is called when configuration is finished
			virtual void weight_vector_bound_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			const_layer_smart_ptr layer_schema;
			cuda_running_configuration_const_smart_ptr cuda_config;

		private:
			weight_vector_bound_cuda(const weight_vector_bound_cuda&);
			weight_vector_bound_cuda& operator =(const weight_vector_bound_cuda&);
		};

		typedef std::tr1::shared_ptr<weight_vector_bound_cuda> weight_vector_bound_cuda_smart_ptr;
		typedef std::map<unsigned int, weight_vector_bound_cuda_smart_ptr> weight_vector_bound_map;
	}
}
