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

#include <iostream>
#include <stdio.h>

//#undef NNFORGE_CUDA_BACKEND_ENABLED

#ifdef NNFORGE_CUDA_BACKEND_ENABLED
#include <nnforge/cuda/cuda.h>
#else
#include <nnforge/plain/plain.h>
#endif
#include "gtsrb_toolset.h"

int main(int argc, char* argv[])
{
	try
	{
		#ifdef NNFORGE_CUDA_BACKEND_ENABLED
		nnforge::cuda::cuda::init();
		#else
		nnforge::plain::plain::init();
		#endif

		#ifdef NNFORGE_CUDA_BACKEND_ENABLED
		gtsrb_toolset gtsrb(nnforge::factory_generator::ptr(new nnforge::cuda::factory_generator_cuda()));
		#else
		gtsrb_toolset gtsrb(nnforge::factory_generator::ptr(new nnforge::plain::factory_generator_plain()));
		#endif

		try
		{
			if (gtsrb.parse(argc, argv))
				gtsrb.do_action();
		}
		catch (const std::exception& e)
		{
			std::cout << "Exception caught: " << e.what() << std::endl;
			return 1;
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Exception caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
