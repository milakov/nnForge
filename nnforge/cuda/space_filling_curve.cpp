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

#include "space_filling_curve.h"

#include <algorithm>

#include "../neural_network_exception.h"

#include "space_filling_curve_z_order.h"

namespace nnforge
{
	namespace cuda
	{
		space_filling_curve::space_filling_curve()
		{
		}

		space_filling_curve::~space_filling_curve()
		{
		}

		std::tr1::shared_ptr<space_filling_curve> space_filling_curve::get_space_filling_curve(space_filling_curve_type curve_type)
		{
			switch (curve_type)
			{
			case space_filling_curve_type_z_order:
				return std::tr1::shared_ptr<space_filling_curve>(new space_filling_curve_z_order());
			}

			throw neural_network_exception("Unknown space-filling curve type requested");
		}

		space_filling_curve::tile::tile(
			const std::vector<int>& start_elems,
			const std::vector<int>& end_elems)
			: start_elems(start_elems)
			, end_elems(end_elems)
		{
		}

		bool space_filling_curve::tile::is_point() const
		{
			std::vector<int>::const_iterator end_it = end_elems.begin();
			for(std::vector<int>::const_iterator start_it = start_elems.begin(); start_it != start_elems.end(); ++start_it, ++end_it)
				if ((*end_it - *start_it) != 1)
					return false;

			return true;
		}

		void space_filling_curve::fill_tiling_pattern(
			const std::vector<int>& size_list,
			std::vector<std::vector<int> >& ordered_list) const
		{
			ordered_list.clear();

			std::stack<tile> work_set;

			int size_max = *std::max_element(size_list.begin(), size_list.end());
			int size_aligned = 1;
			while (size_aligned < size_max)
				size_aligned <<= 1;

			work_set.push(tile(std::vector<int>(size_list.size(), 0), std::vector<int>(size_list.size(), size_aligned)));

			std::vector<int> start_elems;
			for(std::vector<int>::const_iterator it = size_list.begin(); it != size_list.end(); ++it)
				start_elems.push_back(size_aligned - *it);

			while (!work_set.empty())
			{
				tile cur_tile = work_set.top();
				work_set.pop();

				if (cur_tile.is_point())
				{
					std::vector<int> new_elem;
					std::vector<int>::const_iterator start_it = start_elems.begin();
					bool is_valid = true;
					for(std::vector<int>::const_iterator it = cur_tile.start_elems.begin(); (it != cur_tile.start_elems.end()) && is_valid; ++it, ++start_it)
					{
						int val = *it - *start_it;
						if (val >= 0)
							new_elem.push_back(val);
						else
							is_valid = false;
					}

					if (is_valid)
						ordered_list.push_back(new_elem);
				}
				else
					split_to_stack(cur_tile, work_set, start_elems);
			}

			int total_size_expected = 1;
			for(std::vector<int>::const_iterator it = size_list.begin(); it != size_list.end(); ++it)
				total_size_expected *= *it;

			if (ordered_list.size() != total_size_expected)
				throw neural_network_exception("Internal error when generating space-filling curve");
		}
	}
}
