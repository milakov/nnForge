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

#include <vector>
#include <array>

namespace nnforge
{
	namespace cuda
	{
		template<int dimension_count>
		class sequential_curve
		{
		public:
			static void fill_pattern(
				const std::array<int, dimension_count>& size_list,
				std::vector<std::array<int, dimension_count> >& ordered_list)
			{
				ordered_list.clear();

				std::array<int, dimension_count> new_elem;
				for(int i = 0; i < dimension_count; ++i)
					new_elem[i] = 0;
				int total_elem_count = 1;
				for(int i = 0; i < dimension_count; ++i)
					total_elem_count *= size_list[i];
				for(int j = 0; j < total_elem_count; ++j)
				{
					ordered_list.push_back(new_elem);
					for(int i = 0; i < dimension_count; ++i)
					{
						new_elem[i]++;
						if (new_elem[i] >= size_list[i])
							new_elem[i] = 0;
						else
							break;
					}
				}
			}
		};
	}
}
