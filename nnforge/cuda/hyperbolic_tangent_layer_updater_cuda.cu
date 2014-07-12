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

#include "hyperbolic_tangent_layer_updater_cuda.h"

#include <cuda_runtime.h>

#include "../hyperbolic_tangent_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include "util_cuda.h"

static __forceinline__ __device__ float hyperbolic_tangent(
	float x,
	float hyperbolic_tangent_steepness2,
	float hyperbolic_tangent_major_multiplier)
{
	float y = __expf(x * hyperbolic_tangent_steepness2);
	return __fdividef(y - 1.0F, y + 1.0F) * hyperbolic_tangent_major_multiplier;
}

__global__ void hyperbolic_tangent_upd_kernel(
	const float4 * __restrict input,
	float4 * __restrict output,
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

static __forceinline__ __device__ float hyperbolic_tangent_deriviative(
	float x,
	float hyperbolic_tangent_major_multiplier_reverted,
	float hyperbolic_tangent_steepness3)
{
	float normalized_value = x * hyperbolic_tangent_major_multiplier_reverted;
	return hyperbolic_tangent_steepness3 * (1.0F - (normalized_value * normalized_value));
}

__global__ void hyperbolic_tangent_deriviative_upd_kernel(
	float4 * __restrict errors,
	const float4 * __restrict output_neurons,
	float hyperbolic_tangent_major_multiplier_reverted,
	float hyperbolic_tangent_steepness3,
	int elem_count)
{
	int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
	if (elem_id < elem_count)
	{
		float4 val = output_neurons[elem_id];
		val.x = hyperbolic_tangent_deriviative(val.x, hyperbolic_tangent_major_multiplier_reverted, hyperbolic_tangent_steepness3);
		val.y = hyperbolic_tangent_deriviative(val.y, hyperbolic_tangent_major_multiplier_reverted, hyperbolic_tangent_steepness3);
		val.z = hyperbolic_tangent_deriviative(val.z, hyperbolic_tangent_major_multiplier_reverted, hyperbolic_tangent_steepness3);
		val.w = hyperbolic_tangent_deriviative(val.w, hyperbolic_tangent_major_multiplier_reverted, hyperbolic_tangent_steepness3);
		float4 current_error = errors[elem_id];
		current_error.x *= val.x;
		current_error.y *= val.y;
		current_error.z *= val.z;
		current_error.w *= val.w;
		errors[elem_id] = current_error;
	}
}

namespace nnforge
{
	namespace cuda
	{
		hyperbolic_tangent_layer_updater_cuda::hyperbolic_tangent_layer_updater_cuda()
		{
		}

		hyperbolic_tangent_layer_updater_cuda::~hyperbolic_tangent_layer_updater_cuda()
		{
		}

		void hyperbolic_tangent_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("hyperbolic_tangent_layer_updater_cuda is not able to run using offset");

			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			hyperbolic_tangent_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*input_neurons_buffer,
				*output_neurons_buffer,
				hyperbolic_tangent_steepness2,
				hyperbolic_tangent_major_multiplier,
				elem_count);
		}

		void hyperbolic_tangent_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count)
		{
			int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			hyperbolic_tangent_deriviative_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*output_errors_buffer,
				*output_neurons_buffer,
				hyperbolic_tangent_major_multiplier_reverted,
				hyperbolic_tangent_steepness3,
				elem_count);
		}

		bool hyperbolic_tangent_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}

		void hyperbolic_tangent_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const hyperbolic_tangent_layer> layer_derived = nnforge_dynamic_pointer_cast<const hyperbolic_tangent_layer>(layer_schema);

			hyperbolic_tangent_steepness2 = layer_derived->steepness * 2.0F;
			hyperbolic_tangent_major_multiplier = layer_derived->major_multiplier;
			hyperbolic_tangent_steepness3 = layer_derived->steepness * layer_derived->major_multiplier;
			hyperbolic_tangent_major_multiplier_reverted = 1.0F / layer_derived->major_multiplier;
		}
	}
}
