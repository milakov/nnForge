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

#include "space_filling_curve.h"
#include "space_filling_curve_tile.h"

namespace nnforge
{
	namespace cuda
	{
		template<int dimension_count>
		class split_policy_z_order
		{
		public:
			static void split_to_stack(
				const space_filling_curve_tile<dimension_count>& tile_to_split,
				std::stack<space_filling_curve_tile<dimension_count> >& st,
				const std::array<int, dimension_count>& start_elems)
			{
				std::array<std::array<int, 3>, dimension_count> boundary_elems;
				for(int i = 0; i < dimension_count; ++i)
				{
					boundary_elems[i][0] = tile_to_split.start_elems[i];
					boundary_elems[i][1] = (tile_to_split.start_elems[i] + tile_to_split.end_elems[i]) >> 1;
					boundary_elems[i][2] = tile_to_split.end_elems[i];
				}

				space_filling_curve_tile<dimension_count> new_elem;
				for (int compacted_index = (1 << dimension_count) - 1; compacted_index >= 0; --compacted_index)
				{
					bool is_valid = true;
					int index_reduced = compacted_index;
					for(int i = 0; (i < dimension_count) && is_valid; ++i, index_reduced >>= 1)
					{
						int start_index = index_reduced & 1;
						int start_elem = boundary_elems[i][start_index];
						int end_elem = boundary_elems[i][start_index + 1];

						if (end_elem > start_elems[i])
						{
							new_elem.start_elems[i] = start_elem;
							new_elem.end_elems[i] = end_elem;
						}
						else
							is_valid = false;
					}

					if (is_valid)
						st.push(new_elem);
				}
			}
		};
	}
}
