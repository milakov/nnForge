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

#include "cuda_event.h"

#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		cuda_event::cuda_event(bool with_timing)
			: event(0)
		{
			cuda_safe_call(cudaEventCreateWithFlags(&event, with_timing ? 0 : cudaEventDisableTiming));
		}

		cuda_event::~cuda_event()
		{
			if (event)
				cudaEventDestroy(event);
		}

		cuda_event::operator cudaEvent_t ()
		{
			return event;
		}
	}
}
