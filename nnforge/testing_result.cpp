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

#include "testing_result.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	testing_result::testing_result()
		: entry_count(0)
		, flops(0.0F)
		, time_to_complete_seconds(0.0F)
	{
	}

	testing_result::testing_result(unsigned int neuron_count)
		: entry_count(0)
		, flops(0.0F)
		, time_to_complete_seconds(0.0F)
	{
		cumulative_mse_list.resize(neuron_count, 0.0F);
	}

	float testing_result::get_mse() const
	{
		float res = 0.0F;

		std::for_each(cumulative_mse_list.begin(), cumulative_mse_list.end(), res += boost::lambda::_1);

		return res / static_cast<float>(entry_count);
	}

	std::ostream& operator<< (std::ostream& out, const testing_result& val)
	{
		out << (boost::format("MSE %|1$.6f|") % val.get_mse());

		if (val.time_to_complete_seconds != 0.0F)
		{
			float gflops = val.flops / val.time_to_complete_seconds * 1.0e-9F;
			out << (boost::format(" (%|1$.1f| GFLOPs, %|2$.2f| seconds)") % gflops % val.time_to_complete_seconds);
		}

		return out;
	}
}
