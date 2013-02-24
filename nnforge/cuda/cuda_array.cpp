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

#include "cuda_array.h"

#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		cuda_array::cuda_array(
			unsigned int width,
			unsigned int height,
			unsigned int layer_count)
			: arr(0)
		{
			cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			struct cudaExtent extent;
			extent.width = width;
			extent.height = height;
			extent.depth = layer_count;
			cuda_safe_call(cudaMalloc3DArray(&arr, &channel_desc, extent, cudaArrayLayered | cudaArraySurfaceLoadStore));
		}

		cuda_array::cuda_array(
			unsigned int width,
			unsigned int layer_count)
			: arr(0)
		{
			cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			struct cudaExtent extent;
			extent.width = width;
			extent.height = layer_count;
			extent.depth = 0;
			cuda_safe_call(cudaMalloc3DArray(&arr, &channel_desc, extent, cudaArrayLayered | cudaArraySurfaceLoadStore));
		}

		cuda_array::~cuda_array()
		{
			if (arr != 0)
				cudaFreeArray(arr);
		}

		cuda_array::operator cudaArray_const_t () const
		{
			return arr;
		}

		cuda_array::operator cudaArray_t ()
		{
			return arr;
		}

		size_t cuda_array::get_size() const
		{
			struct cudaExtent extent;
			cuda_safe_call(cudaArrayGetInfo(0, &extent, 0, arr));

			size_t res = sizeof(float);
			res *= extent.width;
			if (extent.height)
				res *= extent.height;
			if (extent.depth)
				res *= extent.depth;

			return res;
		}
	}
}
