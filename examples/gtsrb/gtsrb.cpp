#include <iostream>
#include <stdio.h>

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
		gtsrb_toolset gtsrb(nnforge::factory_generator_smart_ptr(new nnforge::cuda::factory_generator_cuda()));
		#else
		gtsrb_toolset gtsrb(nnforge::factory_generator_smart_ptr(new nnforge::plain::factory_generator_plain()));
		#endif

		if (gtsrb.parse(argc, argv))
			gtsrb.do_action();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}

	return 0;
}
