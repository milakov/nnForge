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

namespace nnforge
{
	namespace cuda
	{
		class space_filling_curve_z_order : public space_filling_curve
		{
		public:
			space_filling_curve_z_order();

			virtual ~space_filling_curve_z_order();

		protected:
			virtual void split_to_stack(
				const tile& tile_to_split,
				std::stack<tile>& st,
				const std::vector<int>& start_elems) const;
		};
	}
}
