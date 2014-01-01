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

#include "space_filling_curve_z_order.h"

namespace nnforge
{
	namespace cuda
	{
		space_filling_curve_z_order::space_filling_curve_z_order()
		{
		}

		space_filling_curve_z_order::~space_filling_curve_z_order()
		{
		}

		void space_filling_curve_z_order::split_to_stack(
			const tile& tile_to_split,
			std::stack<tile>& st,
			const std::vector<int>& start_elems) const
		{
			std::vector<std::vector<int> > boundary_elems;
			std::vector<int>::const_iterator end_it = tile_to_split.end_elems.begin();
			for(std::vector<int>::const_iterator start_it = tile_to_split.start_elems.begin(); start_it != tile_to_split.start_elems.end(); ++start_it, ++end_it)
			{
				std::vector<int> new_elem;
				new_elem.push_back(*start_it);
				new_elem.push_back((*start_it + *end_it) >> 1);
				new_elem.push_back(*end_it);

				boundary_elems.push_back(new_elem);
			}

			int dimension_count = start_elems.size();

			for (int compacted_index = (1 << dimension_count) - 1; compacted_index >= 0; --compacted_index)
			{
				std::vector<int> start_bounds;
				std::vector<int> end_bounds;

				bool is_valid = true;
				std::vector<int>::const_iterator start_elems_it = start_elems.begin();
				int index_reduced = compacted_index;
				for(std::vector<std::vector<int> >::const_iterator it = boundary_elems.begin(); (it != boundary_elems.end()) && is_valid; ++it, ++start_elems_it, index_reduced >>= 1)
				{
					int start_index = index_reduced & 1;
					int start_elem = it->at(start_index);
					int end_elem = it->at(start_index + 1);

					if (end_elem > *start_elems_it)
					{
						start_bounds.push_back(start_elem);
						end_bounds.push_back(end_elem);
					}
					else
						is_valid = false;
				}

				if (is_valid)
					st.push(tile(start_bounds, end_bounds));
			}
		}
	}
}
