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

#include "testing_complete_result_set_roc_visualizer.h"

#include <boost/format.hpp>

#include "roc_result.h"

namespace nnforge
{
	testing_complete_result_set_roc_visualizer::testing_complete_result_set_roc_visualizer()
	{
	}

	testing_complete_result_set_roc_visualizer::~testing_complete_result_set_roc_visualizer()
	{
	}

	void testing_complete_result_set_roc_visualizer::dump(
		std::ostream& out,
		const testing_complete_result_set& val) const
	{
		testing_complete_result_set_visualizer::dump(out, val);

		roc_result rr(*val.predicted_output_neuron_value_set, *val.actual_output_neuron_value_set);
		out << ", " << rr;
	}
}
