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

#include "layer_tester_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class sparse_fully_connected_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			sparse_fully_connected_layer_tester_cuda();

			virtual ~sparse_fully_connected_layer_tester_cuda();

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			virtual cuda_linear_buffer_device_smart_ptr get_output_buffer(
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers);

		protected:
			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual void tester_configured();

			virtual void notify_data_custom(const_layer_data_custom_smart_ptr host_data_custom);

		private:
			int feature_map_connection_count;
			int max_column_index_count_per_row;
			int window_size;

			static const int max_input_feature_map_block_size;

			std::pair<int, int> get_input_feature_map_block_size_and_count() const;
		};
	}
}
