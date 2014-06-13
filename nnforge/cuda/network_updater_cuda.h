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

#include "../network_updater.h"
#include "cuda_running_configuration.h"
#include "buffer_cuda_size_configuration.h"
#include "cuda_stream.h"
#include "layer_testing_schema.h"
#include "layer_updater_schema.h"
#include "weight_vector_bound_cuda.h"
#include "error_function_updater_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class network_updater_cuda : public network_updater
		{
		public:
			network_updater_cuda(
				network_schema_smart_ptr schema,
				const_error_function_smart_ptr ef,
				const std::map<unsigned int, float>& layer_to_dropout_rate_map,
				const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
				float weight_decay,
				cuda_running_configuration_const_smart_ptr cuda_config);

			virtual ~network_updater_cuda();

			virtual unsigned int get_max_batch_size() const;

		protected:
			// schema, data and reader are guaranteed to be compatible
			virtual std::vector<testing_result_smart_ptr> actual_update(
				supervised_data_reader& reader,
				const std::vector<network_data_smart_ptr>& learning_rate_vector_list,
				std::vector<network_data_smart_ptr>& data_list);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			network_updater_cuda(const network_updater_cuda&);
			network_updater_cuda& operator =(const network_updater_cuda&);

			void setup_network_cuda();

			void update_buffers_configuration(
				buffer_cuda_size_configuration& buffer_configuration,
				unsigned int updater_entry_count) const;

			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > get_learning_rate(const std::vector<network_data_smart_ptr>& learning_rate_list) const;

			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > get_data(const std::vector<network_data_smart_ptr>& data_list) const;

			void read_data(
				std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data_list,
				std::vector<network_data_smart_ptr>& res,
				cudaStream_t stream_id) const;

			void enqueue_dropout(
				cudaStream_t stream_id,
				const_cuda_linear_buffer_device_smart_ptr random_buffer,
				cuda_linear_buffer_device_smart_ptr target_buffer,
				float dropout_rate,
				unsigned int mask,
				unsigned int elem_count,
				unsigned int offset_in_random_list);

			cuda_running_configuration_const_smart_ptr cuda_config;

			cuda_stream_smart_ptr command_stream;
			cuda_stream_smart_ptr data_stream;

			unsigned int testing_layer_count;
			const_layer_list::const_iterator start_layer_nonempty_weights_iterator;

			const_layer_testing_schema_list testing_schemas;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > testing_schema_data;
			std::vector<layer_tester_cuda_smart_ptr> tester_list;

			const_layer_updater_schema_list updater_schemas;
			std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> > updater_schema_data;
			std::vector<layer_updater_cuda_smart_ptr> updater_list;
			std::vector<std::vector<unsigned int> > incoming_weight_count_per_output_neuron_list_list;

			weight_vector_bound_map weight_vector_bounds;

			const_error_function_updater_cuda_smart_ptr ef_updater;

			static unsigned int max_entry_count_in_single_batch;
		};
	}
}
