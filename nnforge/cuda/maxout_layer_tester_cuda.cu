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

#include "maxout_layer_tester_cuda.h"

#include <cuda_runtime.h>
#include <memory>

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

#include "../maxout_layer.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void maxout_kernel(
			float * __restrict output,
			const float * __restrict input,
			int neuron_count_per_feature_map,
			int input_neuron_count,
			int output_neuron_count,
			int output_feature_map_count,
			int feature_map_subsampling_size,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int output_feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

			if ((neuron_id < neuron_count_per_feature_map) && (output_feature_map_id < output_feature_map_count) && (entry_id < entry_count))
			{
				bool second_item_valid = (entry_id + 1 < entry_count);
				int elem_id = output_feature_map_id * neuron_count_per_feature_map + neuron_id;
				int input_offset1 = entry_id * input_neuron_count + elem_id;
				int input_offset2 = input_offset1 + input_neuron_count;
				float max_val1 = input[input_offset1];
				float max_val2 = 0;
				if (second_item_valid)
					max_val2 = input[input_offset2];
				for(int i = 1; i < feature_map_subsampling_size; ++i)
				{
					input_offset1 += output_neuron_count;
					input_offset2 += output_neuron_count;
					float new_val1 = input[input_offset1];
					float new_val2 = 0;
					if (second_item_valid)
						new_val2 = input[input_offset2];
					max_val1 = max(new_val1, max_val1);
					max_val2 = max(new_val2, max_val2);
				}
				int output_offset = entry_id * output_neuron_count + elem_id;
				output[output_offset] = max_val1;
				if (second_item_valid)
					output[output_offset + output_neuron_count] = max_val2;
			}
		}

		void maxout_layer_tester_cuda::enqueue_forward_propagation(
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
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				(entry_count + 1) >> 1);

			maxout_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				output_elem_count_per_feature_map,
				input_elem_count_per_entry_list[0],
				output_elem_count_per_entry,
				output_configuration_specific.feature_map_count,
				feature_map_subsampling_size,
				entry_count);
		}

		void maxout_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const maxout_layer> layer_derived = std::dynamic_pointer_cast<const maxout_layer>(layer_schema);

			feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
		}
	}
}
