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

#include "cdf_to_pdf_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "../cdf_to_pdf_layer.h"
#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void cdf_to_pdf_kernel(
			float * __restrict output,
			const float * __restrict input,
			float clamp_min,
			float clamp_max,
			int_fastdiv feature_map_segment_length,
			int neuron_count_per_feature_map,
			int feature_map_count,
			int entry_count)
		{
			int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
			int feature_map_id = blockIdx.y * blockDim.y + threadIdx.y;
			int entry_id = blockIdx.z * blockDim.z + threadIdx.z;

			if ((neuron_id < neuron_count_per_feature_map) || (feature_map_id < feature_map_count) || (entry_id < entry_count))
			{
				int offset = (entry_id * feature_map_count + feature_map_id) * neuron_count_per_feature_map + neuron_id;
				float previous_val = ((feature_map_id % feature_map_segment_length) != 0) ? input[offset - neuron_count_per_feature_map] : 0.0F;
				float current_val = input[offset];
				output[offset] = min(max(current_val - previous_val, clamp_min), clamp_max);
			}
		}

		void cdf_to_pdf_layer_tester_cuda::enqueue_forward_propagation(
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
				entry_count);

			cdf_to_pdf_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				clamp_min,
				clamp_max,
				feature_map_segment_length,
				output_elem_count_per_feature_map,
				output_configuration_specific.feature_map_count,
				entry_count);
		}

		void cdf_to_pdf_layer_tester_cuda::tester_configured()
		{
			std::shared_ptr<const cdf_to_pdf_layer> layer_derived = std::dynamic_pointer_cast<const cdf_to_pdf_layer>(layer_schema);

			feature_map_segment_length = layer_derived->feature_map_segment_length;
			clamp_min = layer_derived->clamp_min;
			clamp_max = layer_derived->clamp_max;
		}
	}
}
