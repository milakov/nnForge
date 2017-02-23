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

#include "concat_layer_tester_cuda.h"

#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void concat_kernel(
			float * __restrict output,
			const float * __restrict input,
			int input_neuron_count,
			int output_neuron_count,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int entry_id = blockIdx.y * blockDim.y + threadIdx.y;
			if ((neuron_id < input_neuron_count) && (entry_id < entry_count))
			{
				output[entry_id * output_neuron_count + neuron_id] = input[entry_id * input_neuron_count + neuron_id];
			}
		}

		void concat_layer_tester_cuda::enqueue_forward_propagation(
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
			int offset = 0;
			for(unsigned int i = 0; i < static_cast<unsigned int>(input_configuration_specific_list.size()); ++i)
			{
				int input_neuron_count = input_elem_count_per_entry_list[i];

				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					input_neuron_count,
					entry_count,
					1);

				concat_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					(float *)*output_buffer + offset,
					*input_buffers[i],
					input_neuron_count,
					output_elem_count_per_entry,
					entry_count);

				offset += input_neuron_count;
			}
		}
	}
}
