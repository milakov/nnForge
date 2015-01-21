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

#include "dropout_layer_updater_cuda.h"

#include "../dropout_layer.h"
#include "neural_network_curand_exception.h"
#include "util_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		__global__ void dropout_upd_kernel(
			const float4 * __restrict input,
			float4 * __restrict output,
			const float4 * __restrict uniform_random, // (0.0,1.0]
			float mult,
			float keep_rate,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = input[elem_id];
				float4 rnd = uniform_random[elem_id];
				val.x *= (rnd.x <= keep_rate ? mult : 0.0F);
				val.y *= (rnd.y <= keep_rate ? mult : 0.0F);
				val.z *= (rnd.z <= keep_rate ? mult : 0.0F);
				val.w *= (rnd.w <= keep_rate ? mult : 0.0F);
				output[elem_id] = val;
			}
		}

		__global__ void dropout_backprop_upd_kernel(
			float4 * __restrict errors,
			const float4 * __restrict uniform_random, // (0.0,1.0]
			float mult,
			float keep_rate,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float4 val = errors[elem_id];
				float4 rnd = uniform_random[elem_id];
				val.x *= (rnd.x <= keep_rate ? mult : 0.0F);
				val.y *= (rnd.y <= keep_rate ? mult : 0.0F);
				val.z *= (rnd.z <= keep_rate ? mult : 0.0F);
				val.w *= (rnd.w <= keep_rate ? mult : 0.0F);
				errors[elem_id] = val;
			}
		}

		dropout_layer_updater_cuda::dropout_layer_updater_cuda()
		{
		}

		dropout_layer_updater_cuda::~dropout_layer_updater_cuda()
		{
		}

		void dropout_layer_updater_cuda::enqueue_test(
			unsigned int offset_input_entry_id,
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("dropout_layer_updater_cuda is not able to run using offset");

			if (force_deterministic)
			{
				cuda_util::copy_buffer(*cuda_config, *input_neurons_buffer, *output_neurons_buffer, input_elem_count_per_entry * entry_count, stream_id);
			}
			else
			{
				int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);

				curand_safe_call(curandSetStream(cuda_config->get_curand_generator(), stream_id));
				curand_safe_call(curandGenerateUniform(cuda_config->get_curand_generator(), *additional_buffers[0], elem_count * 4));

				dropout_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*input_neurons_buffer,
					*output_neurons_buffer,
					*additional_buffers[0],
					mult,
					keep_rate,
					elem_count);
			}
		}

		void dropout_layer_updater_cuda::enqueue_backprop(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& data_custom,
			const_cuda_linear_buffer_device_smart_ptr output_neurons_buffer,
			const_cuda_linear_buffer_device_smart_ptr input_neurons_buffer,
			cuda_linear_buffer_device_smart_ptr output_errors_buffer,
			cuda_linear_buffer_device_smart_ptr input_errors_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			std::vector<cuda_memobject_smart_ptr>& dynamic_memobjects,
			unsigned int entry_count,
			bool force_deterministic)
		{
			if (!force_deterministic)
			{
				int elem_count = (input_elem_count_per_entry * entry_count + 3) / 4;
				std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
					*cuda_config,
					elem_count);
				dropout_backprop_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
					*output_errors_buffer,
					*additional_buffers[0],
					mult,
					keep_rate,
					elem_count);
			}
		}

		std::vector<size_t> dropout_layer_updater_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		void dropout_layer_updater_cuda::updater_configured()
		{
			nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
			dropout_rate = layer_derived->dropout_rate;
			keep_rate = 1.0F - dropout_rate;
			mult = 1.0F / keep_rate;
		}

		bool dropout_layer_updater_cuda::is_in_place_backprop() const
		{
			return true;
		}
	}
}
