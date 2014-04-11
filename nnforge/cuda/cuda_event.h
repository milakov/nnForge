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

#pragma once

#include "../nn_types.h"

#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		class cuda_event
		{
		public:
			cuda_event();

			virtual ~cuda_event();

			operator cudaEvent_t ();

		protected:
			cudaEvent_t evnt;

		private:
			cuda_event(const cuda_event&);
			cuda_event& operator =(const cuda_event&);
		};

		typedef nnforge_shared_ptr<cuda_event> cuda_event_smart_ptr;
	}
}
