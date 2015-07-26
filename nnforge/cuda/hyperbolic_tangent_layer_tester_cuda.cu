/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "hyperbolic_tangent_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"

#include "../hyperbolic_tangent_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		__forceinline__ __device__ float hyperbolic_tangent(
			float x,
			float hyperbolic_tangent_steepness2,
			float hyperbolic_tangent_major_multiplier)
		{
			float y = __expf(x * hyperbolic_tangent_steepness2);
			return __fdividef(y - 1.0F, y + 1.0F) * hyperbolic_tangent_major_multiplier;
		}

		__global__ void hyperbolic_tangent_kernel(
			float4 * __restrict output,
			const float4 * __restrict input,
			float hyperbolic_tangent_steepness2,
			float hyperbolic_tangent_major_multiplier,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input[elem_id];
				val.x = hyperbolic_tangent(val.x, hyperbolic_tangent_steepness2, hyperbolic_tangent_major_multiplier);
				val.y = hyperbolic_tangent(val.y, hyperbolic_tangent_steepness2, hyperbolic_tangent_major_multiplier);
				val.z = hyperbolic_tangent(val.z, hyperbolic_tangent_steepness2, hyperbolic_tangent_major_multiplier);
				val.w = hyperbolic_tangent(val.w, hyperbolic_tangent_steepness2, hyperbolic_tangent_major_multiplier);
				output[elem_id] = val;
			}
		}

		hyperbolic_tangent_layer_tester_cuda::hyperbolic_tangent_layer_tester_cuda()
		{
		}

		hyperbolic_tangent_layer_tester_cuda::~hyperbolic_tangent_layer_tester_cuda()
		{
		}

		void hyperbolic_tangent_layer_tester_cuda::enqueue_forward_propagation(
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
			int elem_count = (output_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			hyperbolic_tangent_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_buffer,
				*input_buffers[0],
				hyperbolic_tangent_steepness2,
				hyperbolic_tangent_major_multiplier,
				elem_count);
		}

		void hyperbolic_tangent_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const hyperbolic_tangent_layer> layer_derived = nnforge_dynamic_pointer_cast<const hyperbolic_tangent_layer>(layer_schema);

			hyperbolic_tangent_steepness2 = layer_derived->steepness * 2.0F;
			hyperbolic_tangent_major_multiplier = layer_derived->scale;
		}

		int hyperbolic_tangent_layer_tester_cuda::get_input_index_layer_can_write() const
		{
			return 0;
		}
	}
}
