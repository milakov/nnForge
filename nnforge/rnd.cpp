#include "rnd.h"

#include <boost/chrono/system_clocks.hpp>
#include <ctime>

namespace nnforge
{
	random_generator rnd::get_random_generator()
	{
		unsigned long seed_value = static_cast<unsigned long>(std::time(0));

#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
		boost::chrono::steady_clock::time_point tp = boost::chrono::high_resolution_clock::now();
		unsigned long add_on = static_cast<unsigned long>(tp.time_since_epoch().count());
		seed_value += add_on;
#endif

		return random_generator(seed_value);
	}
}
