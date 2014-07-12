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

#include "nn_types.h"
#include "error_function.h"

#include <vector>
#include <ostream>

namespace nnforge
{
	class testing_result
	{
	public:
		testing_result(const_error_function_smart_ptr ef);

		float get_error() const;

		const_error_function_smart_ptr ef;

		float flops;
		float time_to_complete_seconds;

		void add_error(float sample_error);

		void add_error(
			double cumulative_error,
			unsigned int entry_count);

		unsigned int get_entry_count() const;

		void init(
			double cumulative_error,
			unsigned int entry_count);

	private:
		double cumulative_error;
		unsigned int entry_count;

		testing_result();
	};

	std::ostream& operator<< (std::ostream& out, const testing_result& val);

	typedef nnforge_shared_ptr<testing_result> testing_result_smart_ptr;
}
