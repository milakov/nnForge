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

#include "../hessian_calculator.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_stream.h"
#include "layer_testing_schema.h"
#include "layer_hessian_schema.h"

namespace nnforge
{
	namespace cuda
	{
		class hessian_calculator_cuda : public hessian_calculator
		{
		public:
			hessian_calculator_cuda(
				network_schema_smart_ptr schema,
				cuda_running_configuration_const_smart_ptr cuda_config);

			virtual ~hessian_calculator_cuda();

		protected:
			// schema, data and reader are guaranteed to be compatible
			virtual network_data_smart_ptr actual_get_hessian(
				unsupervised_data_reader& reader,
				network_data_smart_ptr data,
				unsigned int hessian_entry_to_process_count);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			hessian_calculator_cuda(const hessian_calculator_cuda&);
			hessian_calculator_cuda& operator =(const hessian_calculator_cuda&);

			void setup_network_cuda();

			void update_buffers_configuration(buffer_cuda_size_configuration& buffer_configuration) const;

			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > enqueue_get_data(
				network_data_smart_ptr data,
				cudaStream_t stream_id) const;

			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > enqueue_get_data_squared(
				std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > data,
				cudaStream_t stream_id) const;

			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > enqueue_get_hessian(
				network_data_smart_ptr data_use_schema_only,
				cudaStream_t stream_id) const;

			void enqueue_average_hessian(
				std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& hessian_data,
				float hessian_entry_to_process_count,
				cudaStream_t stream_id) const;

			void enqueue_read_hessian(
				std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& hessian_data,
				network_data_smart_ptr res,
				cudaStream_t stream_id) const;

			cuda_running_configuration_const_smart_ptr cuda_config;

			cuda_stream_smart_ptr command_stream;
			cuda_stream_smart_ptr data_stream;

			unsigned int testing_layer_count;
			const_layer_list::const_iterator start_layer_nonempty_weights_iterator;

			const_layer_testing_schema_list testing_schemas;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > testing_schema_data;
			std::vector<layer_tester_cuda_smart_ptr> tester_list;

			const_layer_hessian_schema_list hessian_schemas;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > hessian_schema_data;
			std::vector<layer_hessian_cuda_smart_ptr> hessian_list;
		};
	}
}
