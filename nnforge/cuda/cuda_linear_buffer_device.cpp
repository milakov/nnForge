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

#include "cuda_linear_buffer_device.h"

#include "neural_network_cuda_exception.h"
#include "cuda_util.h"

#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		cuda_linear_buffer_device::cuda_linear_buffer_device(size_t size)
			: cuda_linear_buffer()
			, buf(0)
			, size(0)
		{
			cuda_safe_call(cudaMalloc(&buf, cuda_util::get_float4_aligned_buffer_size(size)));
			this->size = size;
		}

		cuda_linear_buffer_device::cuda_linear_buffer_device(
			const void * src,
			size_t size,
			cudaStream_t stream_id)
		{
			cuda_safe_call(cudaMalloc(&buf, cuda_util::get_float4_aligned_buffer_size(size)));
			this->size = size;
			if (stream_id)
			{
				cuda_safe_call(cudaMemcpyAsync(buf, src, size, cudaMemcpyHostToDevice, stream_id));
			}
			else
			{
				cuda_safe_call(cudaMemcpy(buf, src, size, cudaMemcpyHostToDevice));
			}
		}

		cuda_linear_buffer_device::~cuda_linear_buffer_device()
		{
			if (buf)
				cudaFree(buf);
		}

		void * cuda_linear_buffer_device::get_buf()
		{
			return buf;
		}

		const void * cuda_linear_buffer_device::get_buf() const
		{
			return buf;
		}

		size_t cuda_linear_buffer_device::get_size() const
		{
			return size;
		}
	}
}
