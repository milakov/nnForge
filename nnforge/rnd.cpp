#include "rnd.h"

#include <boost/chrono/system_clocks.hpp>
#include <ctime>

namespace nnforge
{
	random_generator rnd::get_random_generator()
	{
		unsigned long seed = static_cast<unsigned long>(std::time(0));

#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
		boost::chrono::steady_clock::time_point tp = boost::chrono::high_resolution_clock::now();
		unsigned long add_on = static_cast<unsigned long>(tp.time_since_epoch().count());
		seed += add_on;
#endif
		return get_random_generator(seed);
	}

	random_generator rnd::get_random_generator(unsigned long seed)
	{
		return random_generator(seed);
	}
}
