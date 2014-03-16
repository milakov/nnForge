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
	testing_result::testing_result(const_error_function_smart_ptr ef)
		: entry_count(0)
		, ef(ef)
		, flops(0.0F)
		, time_to_complete_seconds(0.0F)
		, cumulative_error(0.0)
	{
	}

	float testing_result::get_error() const
	{
		return static_cast<float>(static_cast<double>(cumulative_error) / static_cast<double>(entry_count));
	}

	void testing_result::add_error(
		const float * actual_values,
		const float * predicted_values,
		unsigned int neuron_count)
	{
		++entry_count;
		cumulative_error += static_cast<double>(ef->calculate_error(actual_values, predicted_values, neuron_count));
	}

	unsigned int testing_result::get_entry_count() const
	{
		return entry_count;
	}

	void testing_result::init(
		double cumulative_error,
		unsigned int entry_count)
	{
		this->cumulative_error = cumulative_error;
		this->entry_count = entry_count;
	}

	std::ostream& operator<< (std::ostream& out, const testing_result& val)
	{
		out << val.ef->get_function_name();
		out << (boost::format(" %|1$.6f|") % val.get_error());

		if (val.time_to_complete_seconds != 0.0F)
		{
			float gflops = val.flops / val.time_to_complete_seconds * 1.0e-9F;
			out << (boost::format(" (%|1$.1f| GFLOPs, %|2$.2f| seconds)") % gflops % val.time_to_complete_seconds);
		}

		return out;
	}
}
