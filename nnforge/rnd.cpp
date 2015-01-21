#include "rnd.h"

#include <boost/chrono/system_clocks.hpp>
#include <ctime>
#include <limits.h>

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
		unsigned int seed = static_cast<unsigned int>(std::time(0));

#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
		boost::chrono::steady_clock::time_point tp = boost::chrono::high_resolution_clock::now();
		unsigned int add_on = static_cast<unsigned int>(tp.time_since_epoch().count());
		seed += add_on;
#endif

		return seed;
	}
}
