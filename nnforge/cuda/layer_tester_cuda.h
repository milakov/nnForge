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

#pragma once

#include "../layer.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_memobject.h"
#include "cuda_linear_buffer_device.h"
#include "../nn_types.h"

#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class layer_tester_cuda
		{
		public:
			virtual ~layer_tester_cuda();

			void configure(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config);

			std::vector<cuda_linear_buffer_device_smart_ptr> allocate_additional_buffers(unsigned int max_entry_count) const;

			void update_buffer_configuration(
				buffer_cuda_size_configuration& buffer_configuration,
				unsigned int tiling_factor) const;

			virtual cuda_linear_buffer_device_smart_ptr get_output_buffer(
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers);

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count) = 0;

			virtual std::vector<const_cuda_linear_buffer_device_smart_ptr> get_data(const_layer_data_smart_ptr host_data) const;

			virtual std::vector<const_cuda_linear_buffer_device_smart_ptr> set_get_data_custom(const_layer_data_custom_smart_ptr host_data);

		protected:
			layer_tester_cuda();

			// The method is called when configuration is finished
			virtual void tester_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const;

			virtual std::vector<unsigned int> get_linear_addressing_through_texture_per_entry() const;

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const;

			virtual void notify_data_custom(const_layer_data_custom_smart_ptr host_data_custom);

			const_layer_smart_ptr layer_schema;
			cuda_running_configuration_const_smart_ptr cuda_config;

			layer_configuration_specific input_configuration_specific;
			layer_configuration_specific output_configuration_specific;

			unsigned int input_elem_count_per_entry;
			unsigned int output_elem_count_per_entry;
			unsigned int input_elem_count_per_feature_map;
			unsigned int output_elem_count_per_feature_map;

		private:
			layer_tester_cuda(const layer_tester_cuda&);
			layer_tester_cuda& operator =(const layer_tester_cuda&);
		};

		typedef nnforge_shared_ptr<layer_tester_cuda> layer_tester_cuda_smart_ptr;
	}
}
