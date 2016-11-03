#include "rnd.h"

#include <chrono>

namespace nnforge
{
	random_generator rnd::get_random_generator()
	{
		return get_random_generator(get_time_dependent_seed());
	}

	random_generator rnd::get_random_generator(unsigned int seed)
	{
		return random_generator(seed);
	}

	unsigned int rnd::get_time_dependent_seed()
	{
		unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		return seed;
	}
}
