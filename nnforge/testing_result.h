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

#include <memory>
#include <vector>
#include <ostream>

namespace nnforge
{
	class testing_result
	{
	public:
		testing_result(bool is_squared_hingle_loss);

		testing_result(
			bool is_squared_hingle_loss,
			unsigned int neuron_count);

		float get_mse() const;

		std::vector<float> cumulative_mse_list;
		unsigned int entry_count;
		bool is_squared_hingle_loss;
		float flops;
		float time_to_complete_seconds;

	private:
		testing_result();
	};

	std::ostream& operator<< (std::ostream& out, const testing_result& val);

	typedef std::tr1::shared_ptr<testing_result> testing_result_smart_ptr;
}
