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
#include <stack>
#include <algorithm>

#include "space_filling_curve_tile.h"
#include "split_policy_z_order.h"

#include "../neural_network_exception.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace cuda
	{
		template<int dimension_count, class split_policy = split_policy_z_order<dimension_count> >
		class space_filling_curve
		{
		public:
			static void fill_pattern(
				const nnforge_array<int, dimension_count>& size_list,
				std::vector<nnforge_array<int, dimension_count> >& ordered_list)
			{
				ordered_list.clear();

				std::stack<space_filling_curve_tile<dimension_count> > work_set;

				int size_max = *std::max_element(size_list.begin(), size_list.end());
				int size_aligned = 1;
				while (size_aligned < size_max)
					size_aligned <<= 1;

				space_filling_curve_tile<dimension_count> whole_space;
				for(int i = 0; i < dimension_count; ++i)
				{
					whole_space.start_elems[i] = 0;
					whole_space.end_elems[i] = size_aligned;
				}
				work_set.push(whole_space);

				nnforge_array<int, dimension_count> start_elems;
				for(int i = 0; i < dimension_count; ++i)
					start_elems[i] = size_aligned - size_list[i];

				while (!work_set.empty())
				{
					space_filling_curve_tile<dimension_count> cur_tile = work_set.top();
					work_set.pop();

					if (cur_tile.is_point())
					{
						nnforge_array<int, dimension_count> new_elem;
						bool is_valid = true;
						for(int i = 0; (i < dimension_count) && is_valid; ++i)
						{
							int val = cur_tile.start_elems[i] - start_elems[i];
							if (val >= 0)
								new_elem[i] = val;
							else
								is_valid = false;
						}

						if (is_valid)
							ordered_list.push_back(new_elem);
					}
					else
						split_policy::split_to_stack(cur_tile, work_set, start_elems);
				}

				int total_size_expected = 1;
				for(int i = 0; i < dimension_count; ++i)
					total_size_expected *= size_list[i];

				if (ordered_list.size() != total_size_expected)
					throw neural_network_exception("Internal error when generating space-filling curve");
			}
		};
	}
}
