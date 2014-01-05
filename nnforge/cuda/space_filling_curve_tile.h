/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include <array>

namespace nnforge
{
	namespace cuda
	{
		template<int dimension_count>
		class space_filling_curve_tile
		{
		public:
			space_filling_curve_tile() {}

			space_filling_curve_tile(
				const std::tr1::array<int, dimension_count>& start_elems,
				const std::tr1::array<int, dimension_count>& end_elems)
				: start_elems(start_elems)
				, end_elems(end_elems)
			{
			}

			bool is_point() const
			{
				for(int i = 0; i < dimension_count; ++i)
					if ((end_elems[i] - start_elems[i]) != 1)
						return false;

				return true;
			}

			std::tr1::array<int, dimension_count> start_elems;
			std::tr1::array<int, dimension_count> end_elems;
		};
	}
}
