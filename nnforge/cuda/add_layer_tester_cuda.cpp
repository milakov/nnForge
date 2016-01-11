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

#include "add_layer_tester_cuda.h"

#include "util_cuda.h"
#include "neural_network_cublas_exception.h"

namespace nnforge
{
	namespace cuda
	{
		add_layer_tester_cuda::add_layer_tester_cuda()
		{
		}

		add_layer_tester_cuda::~add_layer_tester_cuda()
		{
		}

		void add_layer_tester_cuda::enqueue_forward_propagation(
			cudaStream_t stream_id,
			cuda_linear_buffer_device::ptr output_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			unsigned int entry_count)
		{
			unsigned int elem_count = output_elem_count_per_entry * entry_count;

			// Copy the first buffer
			if ((const float *)(*input_buffers[0]) != (const float *)(*output_buffer))
			{
				cuda_util::copy_buffer(
					*cuda_config,
					*input_buffers[0],
					*output_buffer,
					elem_count,
					stream_id);
			}

			for(unsigned int i = 0; i < static_cast<unsigned int>(input_configuration_specific_list.size()); ++i)
			{
				cublas_safe_call(cublasSetStream(cuda_config->get_cublas_handle(), stream_id));
				float alpha = 1.0F;
				cublas_safe_call(cublasSaxpy(
					cuda_config->get_cublas_handle(),
					elem_count,
					&alpha,
					*input_buffers[i],
					1,
					*output_buffer,
					1));
			}
		}

		int add_layer_tester_cuda::get_input_index_layer_can_write() const
		{
			return 0;
		}
	}
}
