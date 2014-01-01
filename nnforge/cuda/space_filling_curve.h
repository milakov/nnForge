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
#include <memory>

namespace nnforge
{
	namespace cuda
	{
		enum space_filling_curve_type
		{
			space_filling_curve_type_z_order
		};

		class space_filling_curve
		{
		public:
			space_filling_curve();

			virtual ~space_filling_curve();

			static std::tr1::shared_ptr<space_filling_curve> get_space_filling_curve(space_filling_curve_type curve_type = space_filling_curve_type_z_order);

			void fill_tiling_pattern(
				const std::vector<int>& size_list,
				std::vector<std::vector<int> >& ordered_list) const;

		protected:
			struct tile
			{
				tile(
					const std::vector<int>& start_elems,
					const std::vector<int>& end_elems);

				bool is_point() const;

				std::vector<int> start_elems;
				std::vector<int> end_elems;
			};

			virtual void split_to_stack(
				const tile& tile_to_split,
				std::stack<tile>& st,
				const std::vector<int>& start_elems) const = 0;
		};
	}
}
