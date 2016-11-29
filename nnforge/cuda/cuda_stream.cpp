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

#include "cuda_stream.h"

#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		cuda_stream::cuda_stream()
			: stream(0)
		{
			cuda_safe_call(cudaStreamCreate(&stream));
		}

		cuda_stream::cuda_stream(int priority)
			: stream(0)
		{
			cuda_safe_call(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority));
		}

		cuda_stream::~cuda_stream()
		{
			if (stream)
				cudaStreamDestroy(stream);
		}

		cuda_stream::operator cudaStream_t ()
		{
			return stream;
		}
	}
}
