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

#include "layer_tester_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class max_subsampling_3d_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			max_subsampling_3d_layer_tester_cuda();

			virtual ~max_subsampling_3d_layer_tester_cuda();

			virtual void enqueue_test(
				cudaStream_t stream_id,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
				const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count);

			virtual cuda_linear_buffer_device_smart_ptr get_output_buffer(
				cuda_linear_buffer_device_smart_ptr input_buffer,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers);

		protected:
			virtual void tester_configured();

			virtual std::vector<size_t> get_sizes_of_additional_buffers_per_entry() const;

			virtual std::vector<size_t> get_sizes_of_additional_buffers_fixed() const;

			virtual void fill_additional_buffers(const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers) const;

		private:
			std::vector<unsigned int> subsampling_sizes;
		};
	}
}
