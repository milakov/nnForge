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

#include "../neural_network_exception.h"

#include <cublas_v2.h>

namespace nnforge
{
	namespace cuda
	{
		class neural_network_cublas_exception : public neural_network_exception
		{
		public:
			neural_network_cublas_exception(
				cublasStatus_t error_code,
				const char * filename,
				int line_number);
		};
	}
}

#define cublas_safe_call(callstr) {cublasStatus_t error_code = callstr; if (error_code != CUBLAS_STATUS_SUCCESS) throw nnforge::cuda::neural_network_cublas_exception(error_code, __FILE__, __LINE__);}
